from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys_rl.candidate_policy_experiment import policy_for, rerank_probabilities
from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings
from recsys_rl.history_reward_data import load_npz
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import append_synthetic_items
from recsys_rl.ranked_policy_data import catalog_reward, mixture
from recsys_rl.ranked_policy_experiment import reward_regression
from recsys_rl.ranked_policy_training import evaluate, intervals, quality_passed
from recsys_rl.ranking_model import emit
from recsys_rl.recency_data import read_json
from recsys_rl.reward_signal import cluster_interval
from recsys_rl.shortlist_mass_data import build_final, load_context, prepare, verify


def probabilities(arrays: dict, reference: np.ndarray, reward: np.ndarray, descriptor: dict) -> np.ndarray:
    if descriptor["mode"] == "truncate":
        return rerank_probabilities(arrays, reference, reward, descriptor)
    if descriptor["mode"] != "preserve":
        raise ValueError("Unknown shortlist normalization mode.")
    base = reference.astype(np.float64)
    if descriptor["reward_weight"] == 0:
        return base.copy()
    candidates = np.argsort(-reference, axis=1, kind="stable")[:, :descriptor["candidate_count"]]
    rows = np.arange(len(candidates))[:, None]
    mass = base[rows, candidates].sum(axis=1, keepdims=True)
    scores = np.log(np.maximum(base[rows, candidates], 1e-30))
    adjusted = reward[rows, candidates].astype(np.float64)
    last = arrays["input_ids"][np.arange(len(candidates)), arrays["lengths"]-1]
    adjusted -= descriptor["repeat_penalty"]*((candidates+3 == last[:, None]) & (last[:, None] >= 3))
    scores += descriptor["reward_weight"]*adjusted
    weights = np.exp(scores-scores.max(axis=1, keepdims=True))
    weights *= mass/weights.sum(axis=1, keepdims=True)
    output = base.copy()
    output[rows, candidates] = weights
    return output


def contract(context: dict, reference: np.ndarray, output: np.ndarray, descriptor: dict) -> dict:
    candidates = np.argsort(-reference, axis=1, kind="stable")[:, :descriptor["candidate_count"]]
    rows = np.arange(len(candidates))[:, None]
    base = reference.astype(np.float64)
    inside = float(np.max(np.abs(output[rows, candidates].sum(axis=1)-base[rows, candidates].sum(axis=1))))
    differences = output-base
    differences[rows, candidates] = 0
    outside = float(np.max(np.abs(differences)))
    identity = bool(np.array_equal(output, base))
    if descriptor["mode"] == "preserve":
        if outside != 0 or inside > context["config"]["evaluation"]["mass_tolerance"]:
            raise ValueError("Shortlist probability mass is not preserved.")
        if descriptor["reward_weight"] == 0 and not identity:
            raise ValueError("Zero-weight rule is not identity.")
    return {"maximum_outside_difference": outside, "maximum_inside_mass_difference": inside, "identical_to_reference": identity}


def recipes(context: dict) -> list[tuple[str, dict]]:
    settings = context["config"]["reranking"]
    common = {"kind": "reranker", "repeat_penalty": settings["repeat_penalty"]}
    values = [(f"preserve-{m}-{a}", {**common, "mode": "preserve", "candidate_count": m, "reward_weight": a})
              for m in settings["candidate_counts"] for a in settings["reward_weights"]]
    values.extend((f"truncate-{v['candidate_count']}-{v['reward_weight']}", {**common, "mode": "truncate", **v})
                  for v in context["config"]["truncated_controls"])
    return values


def rollout(context: dict, arrays: dict, cache: dict, descriptor: dict | None, mode: str) -> tuple[dict, dict]:
    settings = context["config"]["rollout"]
    rng = np.random.default_rng(settings["seed"])
    count = min(settings["contexts"], len(arrays["lengths"]))
    indices = np.sort(rng.choice(len(arrays["lengths"]), count, replace=False))
    uniforms = rng.random((count, settings["steps"]))
    working = {k: arrays[k][indices].copy() for k in ("input_ids", "lengths")}
    policy = new_policy(context)
    actions = np.empty((count, settings["steps"]), dtype=np.int32)
    rewards = np.empty(actions.shape, dtype=np.float32)
    for step in range(settings["steps"]):
        logits = policy_logits(policy, working, context["device"], context["config"]["execution"]["inference_batch_size"])
        p = mixture(context, working, logits).astype(np.float64)
        if descriptor is not None:
            p = probabilities(working, p, cache["reward"][indices], descriptor)
        p /= p.sum(axis=1, keepdims=True)
        if mode == "greedy":
            action = np.argmax(p, axis=1)+3
        elif mode == "sampled":
            action = np.minimum((np.cumsum(p, axis=1) < uniforms[:, step, None]).sum(axis=1), p.shape[1]-1)+3
        else:
            raise ValueError("Unknown rollout mode.")
        actions[:, step] = action
        rewards[:, step] = cache["reward"][indices, action-3]
        working["input_ids"], working["lengths"] = append_synthetic_items(working["input_ids"], working["lengths"], action)
    summary = {"contexts": count, "mean_unique": float(np.mean([len(np.unique(row)) for row in actions])),
               "repeat_step5": float(np.mean(actions[:, -1] == actions[:, -2])),
               "repeat_by_step": [None, *[float(np.mean(actions[:, i] == actions[:, i-1])) for i in range(1, settings["steps"])]],
               "reward_by_step": rewards.mean(axis=0, dtype=np.float64).tolist(),
               "observed_response_history_frozen": True, "real_feedback_measured": False}
    return {"indices": indices, "actions": actions, "reward": rewards}, summary


def save_rollout(context: dict, arrays: dict, cache: dict, scope: str, key: str, descriptor: dict | None, mode: str) -> dict:
    root = context["output"] / "rollouts" / scope
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{key}-{mode}"
    if (root / f"{stem}.json").exists():
        record = read_json(root / f"{stem}.json")
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Shortlist-mass rollout changed.")
        return record
    path = root / f"{stem}.npz"
    if path.exists():
        raise FileExistsError("Preserve incomplete shortlist-mass rollout.")
    value, summary = rollout(context, arrays, cache, descriptor, mode)
    np.savez_compressed(path, **value)
    record = {**summary, "path": str(path), "sha256": sha256(path), "descriptor": descriptor}
    write_json(root / f"{stem}.json", record)
    emit(stage="mass_rollout", scope=scope, policy=key, mode=mode, unique=summary["mean_unique"], repeat_step5=summary["repeat_step5"])
    return record


def rule_run(context: dict, key: str, descriptor: dict, reference: np.ndarray) -> dict:
    root = context["output"] / "rules" / key
    if (root / "summary.json").exists():
        return read_json(root / "summary.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete shortlist-mass evaluation.")
    root.mkdir(parents=True, exist_ok=True)
    cache = context["development_cache"]
    p = probabilities(context["validation"], reference, cache["reward"], descriptor)
    value, _, rankings = evaluate(context, context["validation"], cache["reward"], p, reference=reference)
    invariants = contract(context, reference, p, descriptor)
    np.savez_compressed(root / "validation.npz", rankings=rankings)
    record = {"descriptor": descriptor, "validation": value, "contract": invariants,
              "artifacts": {"validation.npz": sha256(root / "validation.npz")}}
    write_json(root / "summary.json", record)
    emit(stage="mass_rule", rule=key, ndcg=value["ranking"]["metrics"]["ndcg@10"], reward=value["top10_reward"])
    return record


def choose(context: dict, rules: dict, rollouts: dict, reference: dict) -> dict:
    eligible = [key for key, r in rules.items() if r["descriptor"]["mode"] == "preserve" and r["descriptor"]["reward_weight"] > 0
                and quality_passed(context, r, reference, rollouts[key]["sampled"], rollouts["reference"]["sampled"])]
    selected = max(eligible, key=lambda key: rules[key]["validation"]["top10_reward"]) if eligible else None
    descriptor = rules[selected]["descriptor"] if selected else None
    rule_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest() if descriptor else None
    return {"selected_rule": selected, "descriptor": descriptor, "rule_sha256": rule_hash, "eligible_rules": eligible,
            "final_used_for_selection": False}


def behavior_intervals(context: dict, arrays: dict, records: dict) -> dict:
    settings = context["config"]["evaluation"]
    saved = {name: load_npz(Path(value["sampled"]["path"])) for name, value in records.items()}
    outputs = {}
    for baseline in ("reference", "matched_truncated"):
        chosen, other = saved["selected"], saved[baseline]
        if not np.array_equal(chosen["indices"], other["indices"]):
            raise ValueError("Final rollout contexts differ.")
        users = arrays["user_ids"][chosen["indices"]]
        unique = np.asarray([len(np.unique(row)) for row in chosen["actions"]])-np.asarray([len(np.unique(row)) for row in other["actions"]])
        repeat = (chosen["actions"][:, -1] == chosen["actions"][:, -2]).astype(float)-(other["actions"][:, -1] == other["actions"][:, -2])
        outputs[baseline] = {name: cluster_interval(values, users, settings["bootstrap_repeats"], settings["bootstrap_seed"]) for name, values in (("mean_unique", unique), ("repeat_step5", repeat))}
        for value in outputs[baseline].values():
            value["scope"] = "Paired percentile user-cluster bootstrap; equal sampled-query weights."
    return outputs


def final_metrics(context: dict, arrays: dict, cache: dict, selection: dict, final_rollouts: dict) -> tuple[dict, dict]:
    reference = mixture(context, arrays, cache["reference_logits"])
    descriptor = selection["descriptor"]
    evaluations, vectors, rankings, invariants = {}, {}, {}, {}
    for name, rule in (("reference", None), ("selected", descriptor), ("matched_truncated", {**descriptor, "mode": "truncate"})):
        p = reference if rule is None else probabilities(arrays, reference, cache["reward"], rule)
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], p, reference=reference)
        if rule is not None:
            invariants[name] = contract(context, reference, p, rule)
    for name in ("historical_rl", "expanded_rl"):
        p = mixture(context, arrays, cache[f"{name}_logits"])
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], p, reference=reference)
    for name, ranking in control_rankings(arrays, context["catalog"], 20).items():
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], rankings=ranking)
    paired = {name: intervals(context, value, vectors["reference"], arrays["user_ids"]) for name, value in vectors.items() if name != "reference"}
    versus_truncated = intervals(context, vectors["selected"], vectors["matched_truncated"], arrays["user_ids"])
    margin = context["config"]["evaluation"]["ndcg_noninferiority_margin"]*evaluations["reference"]["ranking"]["metrics"]["ndcg@10"]
    criteria = context["config"]["selection"]
    base_sampled, sampled = final_rollouts["reference"]["sampled"], final_rollouts["selected"]["sampled"]
    behavior = sampled["mean_unique"] >= base_sampled["mean_unique"]-criteria["maximum_unique_decrease"] and sampled["repeat_step5"] <= base_sampled["repeat_step5"]+criteria["maximum_sampled_repeat_increase"]
    primary = {"reward_improvement": paired["selected"]["reward"]["ci95"][0] > 0,
               "ndcg_noninferiority": paired["selected"]["ndcg@10"]["ci95"][0] >= -margin,
               "ndcg_strict_improvement": paired["selected"]["ndcg@10"]["ci95"][0] > 0,
               "sampled_point_guards_passed": bool(behavior), "absolute_ndcg_margin": margin}
    primary["static_passed"] = primary["reward_improvement"] and primary["ndcg_noninferiority"]
    primary["all_passed"] = primary["static_passed"] and primary["sampled_point_guards_passed"]
    users = load_npz(context["new_data_dir"] / "users.npz")
    counts = users["early_counts"][np.searchsorted(users["users"], arrays["user_ids"])]
    slices = {}
    for name, mask in (("0-4", counts < 5), ("5-49", (counts >= 5) & (counts < 50)), ("50+", counts >= 50)):
        slices[name] = {"queries": int(mask.sum()), "users": len(np.unique(arrays["user_ids"][mask])),
                        "metrics": {policy: {metric: float(values[metric][mask].mean()) if mask.any() else None for metric in ("ndcg@10", "recall@10", "reward")} for policy, values in vectors.items()}}
    return {"policies": evaluations, "contracts": invariants, "intervals_vs_reference": paired, "selected_vs_matched_truncated": versus_truncated,
            "primary_confirmation": primary, "early_activity_slices": slices, "behavior_intervals": behavior_intervals(context, arrays, final_rollouts)}, rankings


def final_evaluation(context: dict, selection: dict) -> dict:
    root = context["output"] / "final"
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete shortlist-mass final evaluation.")
    root.mkdir()
    opening = {"opened_at_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": sha256(context["output"] / "selection.json")}
    write_json(root / "opening.json", opening)
    emit(stage="mass_final_opening", selected=selection["selected_rule"])
    arrays = load_npz(context["new_data_dir"] / "final.npz")
    cache = {"reward": catalog_reward(context, arrays)}
    descriptors = {"reference": {"kind": "reference"}, "historical_rl": {"kind": "rl", "checkpoint": str(context["historical_checkpoint"]), "checkpoint_sha256": sha256(context["historical_checkpoint"])}, "expanded_rl": context["expanded_descriptor"]}
    for name, descriptor in descriptors.items():
        field = "reference_logits" if name == "reference" else f"{name}_logits"
        cache[field] = policy_logits(policy_for(context, descriptor), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    rollouts = {name: {mode: save_rollout(context, arrays, cache, "final", name, descriptor, mode) for mode in context["config"]["rollout"]["modes"]}
                for name, descriptor in (("reference", None), ("selected", selection["descriptor"]), ("matched_truncated", {**selection["descriptor"], "mode": "truncate"}))}
    metrics, rankings = final_metrics(context, arrays, cache, selection, rollouts)
    regression, observed = reward_regression(context, arrays, cache["reward"])
    for name, value in (("cache", cache), ("rankings", rankings), ("observed", observed)):
        np.savez_compressed(root / f"{name}.npz", **value)
    result = {**metrics, "reward_model": regression, "rollouts": rollouts, "opening": opening,
              "queries": len(arrays["lengths"]), "users": len(np.unique(arrays["user_ids"])), "observed_pairs": len(observed["rewards"]),
              "scope": "Remaining users including less active and late-arriving users; not the previous early-active population.",
              "final_used_for_selection": False, "causal_user_effect_measured": False,
              "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}}
    write_json(root / "result.json", result)
    return result


def run_series(context: dict, preparation: dict) -> dict:
    root = context["output"]
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    started = time.perf_counter()
    arrays, cache = context["validation"], context["development_cache"]
    reference = mixture(context, arrays, cache["reference_logits"])
    baseline, _, _ = evaluate(context, arrays, cache["reward"], reference, reference=reference)
    rules = {}
    rollouts = {"reference": {"sampled": save_rollout(context, arrays, cache, "development", "reference", None, "sampled")}}
    for key, descriptor in recipes(context):
        rules[key] = rule_run(context, key, descriptor, reference)
        rollouts[key] = {"sampled": save_rollout(context, arrays, cache, "development", key, descriptor, "sampled")}
        if descriptor["mode"] == "preserve" and descriptor["reward_weight"] == 0:
            identity, original = load_npz(Path(rollouts[key]["sampled"]["path"])), load_npz(Path(rollouts["reference"]["sampled"]["path"]))
            if any(not np.array_equal(v, original[k]) for k, v in identity.items()):
                raise ValueError("Zero-weight sampled rollout is not reference identity.")
    selection = choose(context, rules, rollouts, baseline)
    for key in ("reference", selection["selected_rule"]):
        if key:
            descriptor = None if key == "reference" else selection["descriptor"]
            rollouts[key]["greedy"] = save_rollout(context, arrays, cache, "development", key, descriptor, "greedy")
    selection["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "selection.json", selection)
    emit(stage="mass_selection", **selection)
    final = final_evaluation(context, selection) if selection["selected_rule"] else None
    verify(context)
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "preparation": preparation,
              "reference": baseline, "rules": rules, "rollouts": rollouts, "selection": selection, "final": final,
              "final_opened": final is not None, "series_seconds": time.perf_counter()-started, "new_training_runs": 0, "pdf_changed": False}
    write_json(root / "result.json", result)
    emit(stage="shortlist_mass_complete", rules=len(rules), final_opened=final is not None, primary=final["primary_confirmation"] if final else None)
    return result


def check_rollouts(context: dict, arrays: dict, cache: dict, records: dict) -> int:
    count = 0
    for modes in records.values():
        for mode, entry in modes.items():
            values, summary = rollout(context, arrays, cache, entry["descriptor"], mode)
            saved = load_npz(Path(entry["path"]))
            if sha256(Path(entry["path"])) != entry["sha256"] or any(not np.array_equal(v, saved[k]) for k, v in values.items()) or summary != {k: v for k, v in entry.items() if k not in ("path", "sha256", "descriptor")}:
                raise ValueError("Shortlist-mass rollout replay differs.")
            count += 1
    return count


def audit(context: dict) -> dict:
    root = context["output"]
    result = read_json(root / "result.json")
    tolerance = context["config"]["evaluation"]["forward_tolerance"]
    rebuilt = build_final(context)
    for name, arrays in rebuilt.items():
        saved = load_npz(context["new_data_dir"] / f"{name}.npz")
        if set(arrays) != set(saved) or any(not np.array_equal(v, saved[k]) for k, v in arrays.items()):
            raise ValueError("Remaining population or raw contexts differ.")
    arrays, cache = context["validation"], context["development_cache"]
    reward = catalog_reward(context, arrays)
    logits = policy_logits(new_policy(context), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    maximum = max(float(np.max(np.abs(reward-cache["reward"]))), float(np.max(np.abs(logits-cache["reference_logits"]))))
    if maximum > tolerance:
        raise ValueError("Development reference/reward cache differs.")
    regression, _ = reward_regression(context, arrays, cache["reward"])
    if regression != context["previous_report"]["cache"]["development_reward_model"]:
        raise ValueError("Direct development reward differs.")
    reference = mixture(context, arrays, cache["reference_logits"])
    baseline, _, _ = evaluate(context, arrays, cache["reward"], reference, reference=reference)
    if baseline != result["reference"] or list(result["rules"]) != [key for key, _ in recipes(context)]:
        raise ValueError("Shortlist-mass baseline or recipe budget differs.")
    for key, descriptor in recipes(context):
        record = result["rules"][key]
        path = root / "rules" / key / "validation.npz"
        if record["descriptor"] != descriptor or sha256(path) != record["artifacts"]["validation.npz"]:
            raise ValueError("Shortlist-mass rule changed.")
        p = probabilities(arrays, reference, cache["reward"], descriptor)
        metrics, _, ranking = evaluate(context, arrays, cache["reward"], p, reference=reference)
        if metrics != record["validation"] or contract(context, reference, p, descriptor) != record["contract"] or not np.array_equal(ranking, load_npz(path)["rankings"]):
            raise ValueError("Shortlist-mass metrics or invariants differ.")
    rollout_count = check_rollouts(context, arrays, cache, result["rollouts"])
    for key, descriptor in recipes(context):
        if descriptor["mode"] == "preserve" and descriptor["reward_weight"] == 0:
            identity = load_npz(Path(result["rollouts"][key]["sampled"]["path"]))
            original = load_npz(Path(result["rollouts"]["reference"]["sampled"]["path"]))
            if any(not np.array_equal(v, original[k]) for k, v in identity.items()):
                raise ValueError("Zero-weight action identity differs.")
    selection = choose(context, result["rules"], result["rollouts"], result["reference"])
    if selection != {k: v for k, v in result["selection"].items() if k != "recorded_at_utc"}:
        raise ValueError("Shortlist-mass selection differs.")
    if result["final"]:
        location = root / "final"
        final = read_json(location / "result.json")
        if final != result["final"] or final["opening"]["selection_sha256"] != sha256(root / "selection.json") or final["opening"]["opened_at_utc"] <= result["selection"]["recorded_at_utc"]:
            raise ValueError("Shortlist-mass final opening differs.")
        for name, expected in final["artifacts"].items():
            if sha256(location / name) != expected:
                raise ValueError("Shortlist-mass final artifact changed.")
        arrays, cache = rebuilt["final"], load_npz(location / "cache.npz")
        if np.max(np.abs(catalog_reward(context, arrays)-cache["reward"])) > tolerance:
            raise ValueError("Shortlist-mass final reward differs.")
        descriptors = {"reference": {"kind": "reference"}, "historical_rl": {"kind": "rl", "checkpoint": str(context["historical_checkpoint"]), "checkpoint_sha256": sha256(context["historical_checkpoint"])}, "expanded_rl": context["expanded_descriptor"]}
        for name, descriptor in descriptors.items():
            field = "reference_logits" if name == "reference" else f"{name}_logits"
            logits = policy_logits(policy_for(context, descriptor), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
            difference = float(np.max(np.abs(logits-cache[field])))
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise ValueError("Shortlist-mass final policy differs.")
        rollout_count += check_rollouts(context, arrays, cache, final["rollouts"])
        metrics, rankings = final_metrics(context, arrays, cache, selection, final["rollouts"])
        regression, observed = reward_regression(context, arrays, cache["reward"])
        stored_rankings, stored_observed = load_npz(location / "rankings.npz"), load_npz(location / "observed.npz")
        if any(metrics[k] != final[k] for k in metrics) or regression != final["reward_model"] or any(not np.array_equal(v, stored_rankings[k]) for k, v in rankings.items()) or any(not np.array_equal(v, stored_observed[k]) for k, v in observed.items()):
            raise ValueError("Shortlist-mass final metrics differ.")
    verify(context)
    return {"stage": "shortlist_mass_audit_complete", "rules_checked": len(result["rules"]), "raw_preparation_reproduced": True,
            "maximum_forward_difference": maximum, "mass_and_identity_checks_passed": True,
            "rollouts_replayed": rollout_count, "selection_recalculated": True, "final_recalculated": result["final"] is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a shortlist while preserving the reference probability mass outside it.")
    parser.add_argument("--config", type=Path, default=Path("configs/shortlist-mass.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/shortlist-mass-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shortlist-mass-v1"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config, args.protocol, args.output_dir, args.device, args.resume or args.audit)
    preparation = prepare(context)
    if args.prepare_only:
        emit(stage="shortlist_mass_prepared", users=preparation["selected_users"], queries=preparation["queries"], query_users=preparation["query_users"], final_opened=False)
    elif args.audit:
        emit(**audit(context))
    else:
        run_series(context, preparation)


if __name__ == "__main__":
    main()
