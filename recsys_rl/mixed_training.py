from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import targets_for
from recsys_rl.mixed_data import read_json, verify_mixed_inputs
from recsys_rl.model import seed_everything
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import evaluate_policy, log_probabilities, paired_policy_intervals
from recsys_rl.train import batch_tensors


def target_matrix(arrays: dict) -> np.ndarray:
    targets = [targets_for(arrays, index) - 3 for index in range(len(arrays["lengths"]))]
    output = np.full((len(targets), max(map(len, targets))), -100, dtype=np.int64)
    for index, values in enumerate(targets):
        output[index, :len(values)] = values
    return output


def mixed_statistics(logits: np.ndarray, reward: np.ndarray, reference: np.ndarray, labels: np.ndarray, variant: dict) -> dict:
    logp = log_probabilities(logits)
    probabilities = np.exp(logp)
    expected = float(np.mean(np.sum(probabilities * reward, axis=1)))
    kl = float(np.mean(np.sum(probabilities * (logp - reference), axis=1)))
    valid = labels != -100
    ce = float(np.mean((-np.take_along_axis(logp, labels.clip(min=0), axis=1) * valid).sum(axis=1) / valid.sum(axis=1)))
    return {"expected_raw_reward": expected, "kl_to_reference": kl, "next_item_ce": ce,
            "weighted_loss": -variant["reward_weight"] * expected + variant["kl_beta_base"] * kl + variant["ce_weight"] * ce}


def evaluate_mixed(context: dict, arrays: dict, rewards: dict, reference_logits: np.ndarray, logits: np.ndarray,
                   observed: dict | None = None) -> tuple[dict, dict]:
    rankings = (np.argsort(-logits, axis=1, kind="stable")[:, :20] + 3).astype(np.int32)
    return evaluate_policy(arrays, rankings, context["catalog"], rewards, context["counts"], context["known_users"],
                           log_probabilities(reference_logits), logits=logits,
                           low_support=context["config"]["evaluation"]["low_support_threshold"], observed=observed)


def mixed_intervals(context: dict, vectors: dict, reference: dict, users: np.ndarray) -> dict:
    settings = context["config"]["evaluation"]
    output = paired_policy_intervals(vectors, reference, users, settings["bootstrap_repeats"], settings["bootstrap_seed"])
    for value in output.values():
        value["scope"] = "Percentile user-cluster bootstrap, with equal query weights."
    return output


def audit_mixed_run(context: dict, caches: dict, rewards: dict, baseline_vectors: dict, record: dict, forward: bool) -> None:
    root = context["output"] / "runs" / record["variant"]["name"] / f"seed-{record['training_seed']}"
    if record["epochs_completed"] != context["config"]["training"]["epochs"] or record["selected_epoch"] != record["epochs_completed"]:
        raise ValueError("Mixed run did not use the fixed final epoch.")
    if record["reference_policy_sha256"] != sha256(context["policy_checkpoint"]):
        raise ValueError("Mixed run reference checkpoint differs.")
    for name, artifact in record["artifacts"].items():
        if sha256(root / name) != artifact["sha256"]:
            raise ValueError(f"Mixed run artifact changed: {root}/{name}")
    with np.load(root / "validation.npz", allow_pickle=False) as saved:
        for field in ("user_ids", "timestamps", "target_tokens", "target_offsets"):
            if not np.array_equal(saved[field], context["validation"][field]):
                raise ValueError("Mixed validation population differs.")
        logits = saved["logits"].copy()
        rankings = np.argsort(-logits, axis=1, kind="stable")[:, :20] + 3
        if not np.array_equal(rankings, saved["rankings"]):
            raise ValueError("Mixed rankings differ from logits.")
    evaluation, vectors = evaluate_mixed(context, context["validation"], rewards["validation"], caches["validation"]["reference_logits"], logits, context["observed_validation"])
    if evaluation != record["validation"] or mixed_intervals(context, vectors, baseline_vectors, context["validation"]["user_ids"]) != record["intervals_vs_baseline"]:
        raise ValueError("Mixed metrics or intervals do not reproduce.")
    if read_json(root / "training-history.json")["epochs"] != record["training_history"]:
        raise ValueError("Mixed training history differs.")
    if forward:
        policy = new_policy(context)
        policy.load_state_dict(torch.load(root / "policy.pt", map_location="cpu", weights_only=True), strict=True)
        reproduced = policy_logits(policy, context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
        if not np.allclose(reproduced, logits, atol=context["config"]["execution"]["forward_tolerance"], rtol=0) or not np.array_equal(np.argsort(-reproduced, axis=1, kind="stable")[:, :20] + 3, rankings):
            raise ValueError("Mixed saved policy forward does not reproduce.")


def train_mixed_variant(context: dict, caches: dict, rewards: dict, baseline_vectors: dict, variant: dict, seed: int) -> dict:
    verify_mixed_inputs(context)
    root = context["output"] / "runs" / variant["name"] / f"seed-{seed}"
    if (root / "summary.json").exists():
        record = read_json(root / "summary.json")
        if record["variant"] != variant or record["training_seed"] != seed:
            raise ValueError("Saved mixed variant differs.")
        audit_mixed_run(context, caches, rewards, baseline_vectors, record, False)
        return record
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Incomplete mixed training must be preserved before retrying: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started, settings = time.perf_counter(), context["config"]["training"]
    seed_everything(seed)
    policy = new_policy(context)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    labels = target_matrix(context["train"])
    reference_logits = caches["train"]["reference_logits"]
    reference_logp = log_probabilities(reference_logits)
    reward = rewards["train"]["raw_completion"]
    initial = mixed_statistics(reference_logits, reward, reference_logp, labels, variant)
    history = []
    print(json.dumps({"stage": "mixed_training_started", "variant": variant["name"], "seed": seed, "initial": initial}), flush=True)
    for epoch in range(settings["epochs"]):
        epoch_started = time.perf_counter()
        order = np.random.default_rng(seed + epoch).permutation(len(labels))
        loss_sum = 0.0
        for offset in range(0, len(order), settings["batch_size"]):
            indices = order[offset:offset + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = policy(*batch_tensors(context["train"], indices, context["device"]))
            logp = functional.log_softmax(logits, dim=1)
            probabilities = logp.exp()
            reference = functional.log_softmax(torch.as_tensor(reference_logits[indices], device=context["device"]), dim=1)
            raw = torch.as_tensor(reward[indices], device=context["device"])
            target = torch.as_tensor(labels[indices], device=context["device"])
            valid = target != -100
            ce = (-logp.gather(1, target.clamp_min(0)) * valid).sum(dim=1) / valid.sum(dim=1)
            expected, kl = (probabilities * raw).sum(dim=1), (probabilities * (logp - reference)).sum(dim=1)
            loss = (-variant["reward_weight"] * expected + variant["kl_beta_base"] * kl + variant["ce_weight"] * ce).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite mixed training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
        train_logits = policy_logits(policy, context["train"], context["device"], context["config"]["execution"]["inference_batch_size"])
        row = {"epoch": epoch + 1, "mean_loss_during_updates": loss_sum / len(labels),
               "after_epoch": mixed_statistics(train_logits, reward, reference_logp, labels, variant), "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        write_json(root / "training-history.json", {"epochs": history})
        print(json.dumps({"stage": "mixed_epoch_completed", "variant": variant["name"], "seed": seed, **row}), flush=True)
    torch.save({name: value.detach().cpu() for name, value in policy.state_dict().items()}, root / "policy.pt")
    logits = policy_logits(policy, context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
    evaluation, vectors = evaluate_mixed(context, context["validation"], rewards["validation"], caches["validation"]["reference_logits"], logits, context["observed_validation"])
    np.savez_compressed(root / "validation.npz", logits=logits, rankings=np.argsort(-logits, axis=1, kind="stable")[:, :20] + 3,
                        **{field: context["validation"][field] for field in ("user_ids", "timestamps", "target_tokens", "target_offsets")})
    record = {"variant": variant, "training_seed": seed, "source_commit": context["plan"]["source_commit"],
              "reference_policy_sha256": sha256(context["policy_checkpoint"]), "epochs_completed": len(history), "selected_epoch": len(history),
              "optimizer_steps": len(history) * int(np.ceil(len(labels) / settings["batch_size"])),
              "initial": initial, "training_history": history, "validation": evaluation,
              "intervals_vs_baseline": mixed_intervals(context, vectors, baseline_vectors, context["validation"]["user_ids"]),
              "seconds": time.perf_counter() - started,
              "artifacts": {name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size} for name in ("policy.pt", "training-history.json", "validation.npz")},
              "fresh_holdout_evaluated": False, "historical_test_reevaluated": False}
    audit_mixed_run(context, caches, rewards, baseline_vectors, record, False)
    write_json(root / "summary.json", record)
    print(json.dumps({"stage": "mixed_run_complete", "variant": variant["name"], "seed": seed,
                      "ndcg": evaluation["ranking"]["metrics"]["ndcg@10"], "hybrid": evaluation["top10_proxy_reward"]["hybrid_completion"]}), flush=True)
    del policy, optimizer
    if context["device"].type == "mps":
        torch.mps.empty_cache()
    return record


def mixed_eligibility(context: dict, record: dict, baseline: dict, sft: dict, strong: dict) -> bool:
    e, settings = record["validation"], context["config"]["selection"]
    behavior, base_behavior = e["ranking"]["behavior_at_10"], baseline["ranking"]["behavior_at_10"]
    return (e["ranking"]["metrics"]["ndcg@10"] >= settings["minimum_relative_ndcg"] * max(baseline["ranking"]["metrics"]["ndcg@10"], sft["validation"]["ranking"]["metrics"]["ndcg@10"]) and
            e["top10_proxy_reward"]["hybrid_completion"] > max(baseline["top10_proxy_reward"]["hybrid_completion"], strong["validation"]["top10_proxy_reward"]["hybrid_completion"], sft["validation"]["top10_proxy_reward"]["hybrid_completion"]) and
            behavior["catalog_coverage"] >= settings["minimum_relative_catalog_coverage"] * base_behavior["catalog_coverage"] and
            behavior["top_10_items_share"] <= settings["maximum_relative_top10_concentration"] * base_behavior["top_10_items_share"])


def rollout_eligible(context: dict, entry: dict, baseline: dict) -> bool:
    e, b, s = entry["sampled"]["summary"], baseline["sampled"]["summary"], context["config"]["selection"]
    return (e["steps"][-1]["consecutive_generated_repeat_fraction"] <= b["steps"][-1]["consecutive_generated_repeat_fraction"] + s["maximum_sampled_repeat_increase"] and
            e["mean_unique_items_per_sequence"] >= b["mean_unique_items_per_sequence"] - s["maximum_unique_items_decrease"])
