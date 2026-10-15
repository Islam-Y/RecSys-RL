from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import evaluate_rankings
from recsys_rl.model import seed_everything
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import append_synthetic_items, query_metrics
from recsys_rl.ranked_policy_data import mixture
from recsys_rl.ranking_model import emit
from recsys_rl.recency_data import read_json
from recsys_rl.recency_model import copy_distribution
from recsys_rl.reward_signal import cluster_interval
from recsys_rl.train import batch_tensors


def hard_repeat(arrays: dict, probabilities: np.ndarray) -> np.ndarray:
    output = probabilities.astype(np.float64).copy()
    last = arrays["input_ids"][np.arange(len(output)), arrays["lengths"]-1]
    valid = last >= 3; output[np.flatnonzero(valid), last[valid]-3] = 0
    return output/output.sum(axis=1, keepdims=True)


def evaluate(context: dict, arrays: dict, reward: np.ndarray, probabilities: np.ndarray | None = None,
             rankings: np.ndarray | None = None, reference: np.ndarray | None = None) -> tuple[dict, dict, np.ndarray]:
    if probabilities is not None:
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6): raise ValueError("Invalid mixture policy.")
        rankings = (np.argsort(-probabilities, axis=1, kind="stable")[:, :20]+3).astype(np.int32)
    quality = evaluate_rankings(arrays, rankings, context["catalog"], [5, 10, 20])
    vectors = query_metrics(arrays, rankings)
    vectors["reward"] = reward[np.arange(len(rankings))[:, None], rankings[:, :10]-3].mean(axis=1).astype(np.float64)
    result = {"ranking": quality, "top10_reward": float(vectors["reward"].mean()), "distribution": None}
    if probabilities is not None:
        p = probabilities.astype(np.float64); logp = np.log(np.maximum(p, 1e-30))
        result["distribution"] = {"expected_reward": float(np.mean(np.sum(p*reward, axis=1))),
                                  "entropy": float(np.mean(-np.sum(p*logp, axis=1))),
                                  "kl_to_reference": float(np.mean(np.sum(p*(logp-np.log(np.maximum(reference, 1e-30))), axis=1)))}
    return result, vectors, rankings


def intervals(context: dict, vectors: dict, baseline: dict, users: np.ndarray) -> dict:
    settings = context["config"]["evaluation"]
    result = {n: cluster_interval(vectors[n]-baseline[n], users, settings["bootstrap_repeats"], settings["bootstrap_seed"])
              for n in ("ndcg@10", "recall@10", "reward")}
    for v in result.values(): v["scope"] = "Percentile user-cluster bootstrap; equal query weights."
    return result


def train(context: dict, caches: dict, variant: dict, seed: int) -> dict:
    root = context["output"] / "runs" / variant["name"] / f"seed-{seed}"
    if (root / "summary.json").exists(): return read_json(root / "summary.json")
    if root.exists() and any(root.iterdir()): raise FileExistsError("Preserve incomplete ranked policy run.")
    root.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    settings = context["config"]["training"]; seed_everything(seed)
    policy = new_policy(context); policy.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    arrays = context["train"]; copy = copy_distribution(arrays, len(context["catalog"]["items"]), context["tau"])
    g = np.float32(context["gate"])*(copy.sum(axis=1) > 0).astype(np.float32)
    reference = mixture(context, arrays, caches["train"]["reference_logits"])
    reward = caches["train"]["reward"].copy()
    last = arrays["input_ids"][np.arange(len(g)), arrays["lengths"]-1]
    valid = last >= 3; reward[np.flatnonzero(valid), last[valid]-3] -= variant["repeat_penalty"]
    history = []
    for epoch in range(1, settings["epochs"]+1):
        order = np.random.default_rng(seed+epoch).permutation(len(g)); total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start+settings["batch_size"]]
            logits = policy(*batch_tensors(arrays, indices, context["device"]))
            weight = torch.as_tensor(g[indices, None], device=context["device"])
            p = (1-weight)*functional.softmax(logits, dim=1)+weight*torch.as_tensor(copy[indices], device=context["device"])
            logp = p.clamp_min(settings["log_floor"]).log()
            reference_log = torch.as_tensor(reference[indices], device=context["device"]).clamp_min(settings["log_floor"]).log()
            target = torch.as_tensor(reward[indices], device=context["device"])
            loss = (-(p*target).sum(dim=1)+variant["beta"]*(p*(logp-reference_log)).sum(dim=1)).mean()
            if not torch.isfinite(loss): raise ValueError("Non-finite ranked-policy loss.")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["gradient_clip"]); optimizer.step()
            total += float(loss.detach().cpu())*len(indices)
        history.append({"epoch": epoch, "mean_update_loss": total/len(order)})
        emit(stage="ranked_policy_epoch", variant=variant["name"], seed=seed, **history[-1])
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, root / "policy.pt")
    logits = policy_logits(policy, context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
    probabilities = mixture(context, context["validation"], logits)
    ref = mixture(context, context["validation"], caches["validation"]["reference_logits"])
    evaluation, _, ranking = evaluate(context, context["validation"], caches["validation"]["reward"], probabilities, reference=ref)
    np.savez_compressed(root / "validation.npz", logits=logits, rankings=ranking)
    write_json(root / "training-history.json", {"epochs": history})
    result = {"variant": variant, "seed": seed, "epochs": len(history), "history": history, "validation": evaluation,
              "checkpoint": str(root / "policy.pt"), "seconds": time.perf_counter()-started,
              "artifacts": {p.name: sha256(p) for p in (root / "policy.pt", root / "validation.npz", root / "training-history.json")}}
    write_json(root / "summary.json", result); return result


def rollout(context: dict, cache: dict, run: dict | None, mode: str, hard: bool = False) -> tuple[dict, dict]:
    settings = context["config"]["rollout"]; rng = np.random.default_rng(settings["seed"])
    indices = np.sort(rng.choice(len(context["validation"]["lengths"]), settings["contexts"], replace=False))
    uniforms = rng.random((len(indices), settings["steps"]))
    arrays = {k: context["validation"][k][indices].copy() for k in ("input_ids", "lengths")}
    policy = new_policy(context)
    if run: policy.load_state_dict(torch.load(run["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    actions = np.empty((len(indices), settings["steps"]), dtype=np.int32); rewards = np.empty_like(actions, dtype=np.float32)
    for step in range(settings["steps"]):
        logits = policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        p = mixture(context, arrays, logits).astype(np.float64)
        p = hard_repeat(arrays, p) if hard else p/p.sum(axis=1, keepdims=True)
        if mode == "greedy": action = np.argmax(p, axis=1)+3
        else: action = np.minimum((np.cumsum(p, axis=1) < uniforms[:, step, None]).sum(axis=1), p.shape[1]-1)+3
        actions[:, step] = action; rewards[:, step] = cache["reward"][indices, action-3]
        arrays["input_ids"], arrays["lengths"] = append_synthetic_items(arrays["input_ids"], arrays["lengths"], action)
    summary = {"mean_unique": float(np.mean([len(np.unique(row)) for row in actions])),
               "repeat_step5": float(np.mean(actions[:, -1] == actions[:, -2])),
               "repeat_by_step": [None, *[float(np.mean(actions[:, i] == actions[:, i-1])) for i in range(1, settings["steps"])]],
               "reward_by_step": rewards.mean(axis=0, dtype=np.float64).tolist(),
               "observed_response_history_frozen": True, "real_feedback_measured": False}
    return {"indices": indices, "actions": actions, "reward": rewards}, summary


def quality_passed(context: dict, record: dict, reference: dict, sampled: dict, base_sampled: dict) -> bool:
    criteria = context["config"]["selection"]; quality = record["validation"]["ranking"]; base = reference["ranking"]
    return (quality["metrics"]["ndcg@10"] >= criteria["minimum_relative_ndcg"]*base["metrics"]["ndcg@10"] and
            record["validation"]["top10_reward"] > reference["top10_reward"] and
            quality["behavior_at_10"]["catalog_coverage"] >= criteria["minimum_relative_coverage"]*base["behavior_at_10"]["catalog_coverage"] and
            quality["behavior_at_10"]["top_10_items_share"] <= criteria["maximum_relative_concentration"]*base["behavior_at_10"]["top_10_items_share"] and
            sampled["repeat_step5"] <= base_sampled["repeat_step5"]+criteria["maximum_sampled_repeat_increase"] and
            sampled["mean_unique"] >= base_sampled["mean_unique"]-criteria["maximum_unique_decrease"])


def choose(context: dict, runs: dict, reference: dict, rollouts: dict) -> dict:
    eligible = [runs[f"{v['name']}/42"] for v in context["config"]["variants"]
                if quality_passed(context, runs[f"{v['name']}/42"], reference, rollouts[f"{v['name']}/42"]["sampled"], rollouts["reference/42"]["sampled"])]
    candidate = max(eligible, key=lambda r: r["validation"]["top10_reward"]) if eligible else None
    seeds = [42, *context["config"]["training"]["confirmation_seeds"]]
    checks = [{"seed": s, "passed": quality_passed(context, runs[f"{candidate['variant']['name']}/{s}"], reference,
                                                   rollouts[f"{candidate['variant']['name']}/{s}"]["sampled"], rollouts["reference/42"]["sampled"])}
              for s in seeds if candidate and f"{candidate['variant']['name']}/{s}" in runs]
    passed = candidate is not None and len(checks) == len(seeds) and all(v["passed"] for v in checks)
    return {"screening_challenger": candidate["variant"]["name"] if candidate else None,
            "confirmation": checks, "accepted_on_development": passed,
            "selected_variant": candidate["variant"]["name"] if passed else "reference",
            "checkpoint": {"path": candidate["checkpoint"], "sha256": candidate["artifacts"]["policy.pt"]} if passed else None,
            "final_used_for_selection": False}
