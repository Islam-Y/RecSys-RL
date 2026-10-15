from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recsys_rl.candidate_policy_data import build_groups, load_context, prepare, prepare_cache, training_context, verify
from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings, targets_for
from recsys_rl.history_reward_data import load_npz
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import append_synthetic_items
from recsys_rl.ranked_policy_data import catalog_reward, mixture
from recsys_rl.ranked_policy_experiment import reward_regression
from recsys_rl.ranked_policy_training import evaluate, intervals, quality_passed, train
from recsys_rl.ranking_model import emit
from recsys_rl.recency_data import read_json


def rerank_probabilities(arrays: dict, reference: np.ndarray, reward: np.ndarray, descriptor: dict) -> np.ndarray:
    candidates = np.argsort(-reference, axis=1, kind="stable")[:, :descriptor["candidate_count"]]
    rows = np.arange(len(candidates))[:, None]
    scores = np.log(np.maximum(reference[rows, candidates].astype(np.float64), 1e-30))
    adjusted = reward[rows, candidates].astype(np.float64)
    last = arrays["input_ids"][np.arange(len(candidates)), arrays["lengths"]-1]
    adjusted -= descriptor["repeat_penalty"]*((candidates+3 == last[:, None]) & (last[:, None] >= 3))
    scores += descriptor["reward_weight"]*adjusted
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    output = np.zeros(reference.shape, dtype=np.float64)
    output[rows, candidates] = weights
    return output


def policy_for(context: dict, descriptor: dict):
    policy = new_policy(context)
    if descriptor["kind"] == "rl":
        path = Path(descriptor["checkpoint"])
        if sha256(path) != descriptor["checkpoint_sha256"]:
            raise ValueError("Candidate-policy checkpoint changed.")
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return policy


def distribution(context: dict, arrays: dict, cache: dict, descriptor: dict, logits: np.ndarray | None = None) -> np.ndarray:
    if descriptor["kind"] in ("reference", "reranker"):
        logits = cache["reference_logits"]
    elif descriptor["kind"] == "rl":
        if logits is None:
            logits = policy_logits(policy_for(context, descriptor), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    else:
        raise ValueError("Unknown candidate-policy descriptor.")
    p = mixture(context, arrays, logits)
    return rerank_probabilities(arrays, p, cache["reward"], descriptor) if descriptor["kind"] == "reranker" else p


def rollout(context: dict, cache: dict, descriptor: dict, mode: str) -> tuple[dict, dict]:
    settings = context["config"]["rollout"]
    rng = np.random.default_rng(settings["seed"])
    indices = np.sort(rng.choice(len(context["validation"]["lengths"]), settings["contexts"], replace=False))
    uniforms = rng.random((len(indices), settings["steps"]))
    arrays = {k: context["validation"][k][indices].copy() for k in ("input_ids", "lengths")}
    policy = policy_for(context, descriptor)
    actions = np.empty((len(indices), settings["steps"]), dtype=np.int32)
    rewards = np.empty(actions.shape, dtype=np.float32)
    for step in range(settings["steps"]):
        logits = policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        p = mixture(context, arrays, logits).astype(np.float64)
        if descriptor["kind"] == "reranker":
            p = rerank_probabilities(arrays, p, cache["reward"][indices], descriptor)
        p /= p.sum(axis=1, keepdims=True)
        if mode == "greedy":
            action = np.argmax(p, axis=1)+3
        elif mode == "sampled":
            action = np.minimum((np.cumsum(p, axis=1) < uniforms[:, step, None]).sum(axis=1), p.shape[1]-1)+3
        else:
            raise ValueError("Unknown rollout mode.")
        actions[:, step] = action
        rewards[:, step] = cache["reward"][indices, action-3]
        arrays["input_ids"], arrays["lengths"] = append_synthetic_items(arrays["input_ids"], arrays["lengths"], action)
    summary = {"mean_unique": float(np.mean([len(np.unique(row)) for row in actions])),
               "repeat_step5": float(np.mean(actions[:, -1] == actions[:, -2])),
               "repeat_by_step": [None, *[float(np.mean(actions[:, i] == actions[:, i-1])) for i in range(1, settings["steps"])]],
               "reward_by_step": rewards.mean(axis=0, dtype=np.float64).tolist(),
               "observed_response_history_frozen": True, "real_feedback_measured": False}
    return {"indices": indices, "actions": actions, "reward": rewards}, summary


def save_rollout(context: dict, cache: dict, key: str, descriptor: dict, mode: str) -> dict:
    root = context["output"] / "rollouts"
    root.mkdir(exist_ok=True)
    stem = key.replace("/", "-seed-")+"-"+mode
    summary_path = root / f"{stem}.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if sha256(Path(summary["path"])) != summary["sha256"]:
            raise ValueError("Saved candidate-policy rollout changed.")
        return summary
    path = root / f"{stem}.npz"
    if path.exists():
        raise FileExistsError("Preserve incomplete candidate-policy rollout.")
    arrays, summary = rollout(context, cache, descriptor, mode)
    np.savez_compressed(path, **arrays)
    summary = {**summary, "path": str(path), "sha256": sha256(path), "descriptor": descriptor}
    write_json(summary_path, summary)
    emit(stage="candidate_rollout", policy=key, mode=mode, unique=summary["mean_unique"], repeat_step5=summary["repeat_step5"])
    return summary


def reranker_recipes(context: dict) -> list[tuple[str, dict]]:
    settings = context["config"]["reranking"]
    return [(f"rerank-{m}-{a}", {"kind": "reranker", "candidate_count": m, "reward_weight": a, "repeat_penalty": settings["repeat_penalty"]})
            for m in settings["candidate_counts"] for a in settings["reward_weights"]]


def shortlist_diagnostics(arrays: dict, reference: np.ndarray, descriptor: dict) -> dict:
    indices = np.argsort(-reference, axis=1, kind="stable")[:, :descriptor["candidate_count"]]
    recalls = [float(np.isin(targets_for(arrays, i), row+3).mean()) for i, row in enumerate(indices)]
    return {"candidate_count": descriptor["candidate_count"], "mean_reference_probability_mass": float(reference[np.arange(len(indices))[:, None], indices].sum(axis=1).mean()),
            "target_recall_within_shortlist": float(np.mean(recalls))}


def reranker_run(context: dict, cache: dict, key: str, descriptor: dict, reference: np.ndarray) -> dict:
    root = context["output"] / "reranking" / key
    if (root / "summary.json").exists():
        return read_json(root / "summary.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete reranking evaluation.")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p = rerank_probabilities(context["validation"], reference, cache["reward"], descriptor)
    value, _, ranking = evaluate(context, context["validation"], cache["reward"], p, reference=reference)
    if descriptor["reward_weight"] == 0:
        expected = np.argsort(-reference, axis=1, kind="stable")[:, :20]+3
        if not np.array_equal(ranking, expected):
            raise ValueError("Zero-weight reranking changed reference top-20.")
    np.savez_compressed(root / "validation.npz", rankings=ranking)
    result = {"descriptor": descriptor, "validation": value, "shortlist": shortlist_diagnostics(context["validation"], reference, descriptor),
              "seconds": time.perf_counter()-started, "artifacts": {"validation.npz": sha256(root / "validation.npz")}}
    write_json(root / "summary.json", result)
    emit(stage="candidate_reranker", policy=key, ndcg=value["ranking"]["metrics"]["ndcg@10"], reward=value["top10_reward"])
    return result


def rl_record(context: dict, caches: dict, variant: dict, seed: int) -> dict:
    local_context, local_caches = training_context(context, caches, variant)
    record = train(local_context, local_caches, variant, seed)
    return {**record, "training_contexts": len(local_context["train"]["lengths"]),
            "descriptor": {"kind": "rl", "checkpoint": record["checkpoint"], "checkpoint_sha256": record["artifacts"]["policy.pt"]}}


def choose(context: dict, rerankers: dict, runs: dict, rollouts: dict, reference: dict) -> dict:
    base_sampled = rollouts["reference"]["sampled"]
    def passed(key, record):
        return quality_passed(context, record, reference, rollouts[key]["sampled"], base_sampled)
    eligible_rerankers = [key for key, record in rerankers.items() if record["descriptor"]["reward_weight"] > 0 and passed(key, record)]
    selected_reranker = max(eligible_rerankers, key=lambda key: rerankers[key]["validation"]["top10_reward"]) if eligible_rerankers else None
    eligible_rl = [f"{v['name']}/42" for v in context["config"]["variants"] if f"{v['name']}/42" in runs and passed(f"{v['name']}/42", runs[f"{v['name']}/42"])]
    screening = max(eligible_rl, key=lambda key: runs[key]["validation"]["top10_reward"]) if eligible_rl else None
    checks = []
    if screening:
        name = screening.split("/")[0]
        for seed in (42, *context["config"]["training"]["confirmation_seeds"]):
            key = f"{name}/{seed}"
            if key in runs and key in rollouts:
                checks.append({"seed": seed, "passed": passed(key, runs[key])})
    confirmed = len(checks) == 1+len(context["config"]["training"]["confirmation_seeds"]) and all(c["passed"] for c in checks)
    selected_rl = screening if confirmed else None
    candidates = ([selected_reranker] if selected_reranker else [])+([selected_rl] if selected_rl else [])
    records = {**rerankers, **runs}
    primary = max(candidates, key=lambda key: records[key]["validation"]["top10_reward"]) if candidates else None
    return {"selected_reranker": selected_reranker, "screening_rl": screening, "selected_rl": selected_rl, "rl_confirmation": checks,
            "primary_challenger": primary, "primary_policy": ("reranker" if primary == selected_reranker else "rl") if primary else None,
            "selected_descriptors": {key: records[key]["descriptor"] for key in candidates}, "final_used_for_selection": False}


def fixed_descriptors(context: dict, selection: dict) -> dict:
    values = {"reference": {"kind": "reference"},
              "historical_rl": {"kind": "rl", "checkpoint": str(context["historical_checkpoint"]), "checkpoint_sha256": sha256(context["historical_checkpoint"])}}
    for name, key in (("reranker", "selected_reranker"), ("rl", "selected_rl")):
        selected = selection[key]
        if selected:
            values[name] = selection["selected_descriptors"][selected]
    return values


def final_metrics(context: dict, arrays: dict, cache: dict, selection: dict) -> tuple[dict, dict]:
    reference = mixture(context, arrays, cache["reference_logits"])
    evaluations, vectors, rankings = {}, {}, {}
    for name, descriptor in fixed_descriptors(context, selection).items():
        p = distribution(context, arrays, cache, descriptor, cache.get(f"{name}_logits"))
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], p, reference=reference)
    for name, ranking in control_rankings(arrays, context["catalog"], 20).items():
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], rankings=ranking)
    paired = {name: intervals(context, value, vectors["reference"], arrays["user_ids"]) for name, value in vectors.items() if name != "reference"}
    versus_old = {name: intervals(context, vectors[name], vectors["historical_rl"], arrays["user_ids"]) for name in ("reranker", "rl") if name in vectors}
    margin = context["config"]["evaluation"]["ndcg_noninferiority_margin"]*evaluations["reference"]["ranking"]["metrics"]["ndcg@10"]
    primary_name = selection["primary_policy"]
    primary = {"policy": primary_name, "reward_improvement": paired[primary_name]["reward"]["ci95"][0] > 0,
               "ndcg_noninferiority": paired[primary_name]["ndcg@10"]["ci95"][0] >= -margin, "absolute_ndcg_margin": margin}
    primary["both_passed"] = primary["reward_improvement"] and primary["ndcg_noninferiority"]
    return {"policies": evaluations, "intervals_vs_reference": paired, "intervals_vs_historical_rl": versus_old, "primary_confirmation": primary}, rankings


def final_evaluation(context: dict, selection: dict) -> dict:
    root = context["output"] / "final"
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete candidate-policy final evaluation.")
    root.mkdir()
    opening = {"opened_at_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": sha256(context["output"] / "selection.json")}
    write_json(root / "opening.json", opening)
    emit(stage="candidate_final_opening", primary=selection["primary_challenger"])
    arrays = load_npz(context["new_data_dir"] / "final.npz")
    cache = {"reward": catalog_reward(context, arrays)}
    for name, descriptor in fixed_descriptors(context, selection).items():
        if descriptor["kind"] != "reranker":
            key = "reference_logits" if name == "reference" else f"{name}_logits"
            cache[key] = policy_logits(policy_for(context, descriptor), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    evaluation, rankings = final_metrics(context, arrays, cache, selection)
    regression, observed = reward_regression(context, arrays, cache["reward"])
    for name, value in (("cache", cache), ("rankings", rankings), ("observed", observed)):
        np.savez_compressed(root / f"{name}.npz", **value)
    report = {**evaluation, "reward_model": regression, "opening": opening, "queries": len(arrays["lengths"]),
              "users": len(np.unique(arrays["user_ids"])), "observed_pairs": len(observed["rewards"]),
              "final_used_for_selection": False, "causal_user_effect_measured": False,
              "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}}
    write_json(root / "result.json", report)
    return report


def run_series(context: dict, caches: dict, preparation: dict) -> dict:
    root = context["output"]
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    started = time.perf_counter()
    reference_p = mixture(context, context["validation"], caches["validation"]["reference_logits"])
    reference, _, _ = evaluate(context, context["validation"], caches["validation"]["reward"], reference_p, reference=reference_p)
    rollouts = {"reference": {"sampled": save_rollout(context, caches["validation"], "reference", {"kind": "reference"}, "sampled")}}
    rerankers, runs = {}, {}
    for key, descriptor in reranker_recipes(context):
        rerankers[key] = reranker_run(context, caches["validation"], key, descriptor, reference_p)
        rollouts[key] = {"sampled": save_rollout(context, caches["validation"], key, descriptor, "sampled")}
    for variant in context["config"]["variants"]:
        key = f"{variant['name']}/42"
        runs[key] = rl_record(context, caches, variant, 42)
        rollouts[key] = {"sampled": save_rollout(context, caches["validation"], key, runs[key]["descriptor"], "sampled")}
    selection = choose(context, rerankers, runs, rollouts, reference)
    if selection["screening_rl"]:
        name = selection["screening_rl"].split("/")[0]
        variant = next(v for v in context["config"]["variants"] if v["name"] == name)
        for seed in context["config"]["training"]["confirmation_seeds"]:
            key = f"{name}/{seed}"
            runs[key] = rl_record(context, caches, variant, seed)
            rollouts[key] = {"sampled": save_rollout(context, caches["validation"], key, runs[key]["descriptor"], "sampled")}
    selection = choose(context, rerankers, runs, rollouts, reference)
    for key in ("reference", selection["selected_reranker"], selection["selected_rl"]):
        if key:
            descriptor = {"kind": "reference"} if key == "reference" else {**rerankers, **runs}[key]["descriptor"]
            rollouts[key]["greedy"] = save_rollout(context, caches["validation"], key, descriptor, "greedy")
    selection["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "selection.json", selection)
    emit(stage="candidate_selection", **selection)
    final = final_evaluation(context, selection) if selection["primary_challenger"] else None
    verify(context)
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "preparation": preparation,
              "cache": read_json(root / "cache/manifest.json"), "reference": reference, "rerankers": rerankers, "runs": runs,
              "rollouts": rollouts, "selection": selection, "final": final, "final_opened": final is not None,
              "series_seconds": time.perf_counter()-started, "pdf_changed": False}
    write_json(root / "result.json", result)
    emit(stage="candidate_policy_complete", runs=len(runs), final_opened=final is not None, primary=final["primary_confirmation"] if final else None)
    return result


def audit(context: dict, caches: dict) -> dict:
    result = read_json(context["output"] / "result.json")
    tolerance = context["config"]["evaluation"]["forward_tolerance"]
    reconstructed = build_groups(context)
    for name, value in reconstructed.items():
        saved = load_npz(context["new_data_dir"] / f"{name}.npz")
        if set(value) != set(saved) or any(not np.array_equal(v, saved[k]) for k, v in value.items()):
            raise ValueError("Candidate-policy raw preparation differs.")
    policy = new_policy(context)
    maximum = 0.0
    for split in ("train", "validation"):
        reward = catalog_reward(context, context[split])
        logits = policy_logits(policy, context[split], context["device"], context["config"]["execution"]["inference_batch_size"])
        difference = max(float(np.max(np.abs(reward-caches[split]["reward"]))), float(np.max(np.abs(logits-caches[split]["reference_logits"]))))
        maximum = max(maximum, difference)
        if difference > tolerance:
            raise ValueError("Candidate-policy reference/reward cache mismatch.")
    regression, observed = reward_regression(context, context["validation"], caches["validation"]["reward"])
    saved_observed = load_npz(context["output"] / "cache/observed-validation.npz")
    if regression != result["cache"]["development_reward_model"] or any(not np.array_equal(v, saved_observed[k]) for k, v in observed.items()):
        raise ValueError("Development reward checks differ.")
    reference = mixture(context, context["validation"], caches["validation"]["reference_logits"])
    baseline, _, _ = evaluate(context, context["validation"], caches["validation"]["reward"], reference, reference=reference)
    if baseline != result["reference"]:
        raise ValueError("Candidate-policy reference metrics differ.")
    for key, record in {**result["rerankers"], **result["runs"]}.items():
        descriptor = record["descriptor"]
        if descriptor["kind"] == "reranker":
            root = context["output"] / "reranking" / key
            p = rerank_probabilities(context["validation"], reference, caches["validation"]["reward"], descriptor)
            if shortlist_diagnostics(context["validation"], reference, descriptor) != record["shortlist"]:
                raise ValueError("Shortlist support differs.")
        else:
            root = Path(record["checkpoint"]).parent
            logits = policy_logits(policy_for(context, descriptor), context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
            saved = load_npz(root / "validation.npz")
            difference = float(np.max(np.abs(logits-saved["logits"])))
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise ValueError("Candidate-policy trained forward differs.")
            p = mixture(context, context["validation"], logits)
            variant = record["variant"]
            local_context, _ = training_context(context, caches, variant)
            if record["training_contexts"] != len(local_context["train"]["lengths"]) or record["epochs"] != context["config"]["training"]["epochs"] or read_json(root / "training-history.json")["epochs"] != record["history"]:
                raise ValueError("Candidate-policy training budget differs.")
        for name, expected in record["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Candidate-policy run artifact changed.")
        value, _, ranking = evaluate(context, context["validation"], caches["validation"]["reward"], p, reference=reference)
        if value != record["validation"] or not np.array_equal(ranking, load_npz(root / "validation.npz")["rankings"]):
            raise ValueError("Candidate-policy development evaluation differs.")
    for key, modes in result["rollouts"].items():
        for mode, entry in modes.items():
            arrays, summary = rollout(context, caches["validation"], entry["descriptor"], mode)
            saved = load_npz(Path(entry["path"]))
            if sha256(Path(entry["path"])) != entry["sha256"] or any(not np.array_equal(v, saved[k]) for k, v in arrays.items()) or summary != {k: v for k, v in entry.items() if k not in ("path", "sha256", "descriptor")}:
                raise ValueError("Candidate-policy rollout replay differs.")
    selection = choose(context, result["rerankers"], result["runs"], result["rollouts"], result["reference"])
    if selection != {k: v for k, v in result["selection"].items() if k != "recorded_at_utc"}:
        raise ValueError("Candidate-policy selection differs.")
    if result["final"]:
        root = context["output"] / "final"
        final = read_json(root / "result.json")
        if final != result["final"] or final["opening"]["selection_sha256"] != sha256(context["output"] / "selection.json") or final["opening"]["opened_at_utc"] <= result["selection"]["recorded_at_utc"]:
            raise ValueError("Candidate-policy final opening differs.")
        for name, expected in final["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Candidate-policy final artifact changed.")
        arrays = load_npz(context["new_data_dir"] / "final.npz")
        cache = load_npz(root / "cache.npz")
        if np.max(np.abs(catalog_reward(context, arrays)-cache["reward"])) > tolerance:
            raise ValueError("Final reward cache differs.")
        for name, descriptor in fixed_descriptors(context, selection).items():
            if descriptor["kind"] != "reranker":
                logits = policy_logits(policy_for(context, descriptor), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
                field = "reference_logits" if name == "reference" else f"{name}_logits"
                if np.max(np.abs(logits-cache[field])) > tolerance:
                    raise ValueError("Final policy forward differs.")
        metrics, rankings = final_metrics(context, arrays, cache, selection)
        regression, observed = reward_regression(context, arrays, cache["reward"])
        saved_rankings, saved_observed = load_npz(root / "rankings.npz"), load_npz(root / "observed.npz")
        if any(metrics[k] != final[k] for k in metrics) or regression != final["reward_model"] or any(not np.array_equal(v, saved_rankings[k]) for k, v in rankings.items()) or any(not np.array_equal(v, saved_observed[k]) for k, v in observed.items()):
            raise ValueError("Candidate-policy final metrics differ.")
    verify(context)
    return {"stage": "candidate_policy_audit_complete", "raw_preparation_reproduced": True, "runs_checked": len(result["runs"]),
            "rerankers_checked": len(result["rerankers"]), "rollouts_replayed": sum(len(v) for v in result["rollouts"].values()),
            "maximum_forward_difference": maximum, "selection_recalculated": True, "final_recalculated": result["final"] is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare shortlist reranking with policy optimization on more late contexts.")
    parser.add_argument("--config", type=Path, default=Path("configs/candidate-policy.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/candidate-policy-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/candidate-policy-v1"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config, args.protocol, args.output_dir, args.device, args.resume or args.audit)
    preparation = prepare(context)
    caches = prepare_cache(context)
    if args.prepare_only:
        emit(stage="candidate_policy_prepared", populations=preparation["populations"], small=preparation["small_train_contexts"], expanded=preparation["expanded_train_contexts"], final_opened=False)
    elif args.audit:
        emit(**audit(context, caches))
    else:
        run_series(context, caches, preparation)


if __name__ == "__main__":
    main()
