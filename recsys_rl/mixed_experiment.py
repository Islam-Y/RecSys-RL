from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch

from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings
from recsys_rl.final_evaluation import reward_model
from recsys_rl.mixed_data import load_mixed_context, prepare_fresh_holdout, read_json, verify_mixed_inputs
from recsys_rl.mixed_training import audit_mixed_run, evaluate_mixed, mixed_eligibility, mixed_intervals, rollout_eligible, train_mixed_variant
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import evaluate_policy, log_probabilities, reward_values
from recsys_rl.policy_rollout import generate_rollouts
from recsys_rl.reward import predict_reward
from recsys_rl.reward_catalog import CatalogReward
from recsys_rl.reward_data import attach_rewards, describe_rewards, validate_rewards
from recsys_rl.reward_evaluation import control_predictions, evaluate_reward
from recsys_rl.reward_signal import transformed


def validation_baseline(context: dict, caches: dict, rewards: dict) -> tuple[dict, dict]:
    return evaluate_mixed(context, context["validation"], rewards["validation"], caches["validation"]["reference_logits"],
                          caches["validation"]["reference_logits"], context["observed_validation"])


def choose_mixed(context: dict, runs: dict, baseline: dict, rollouts: dict) -> dict | None:
    candidates = []
    seed = context["config"]["training_seed"]
    for name in context["config"]["selection"]["candidate_names"]:
        record = runs[f"{name}/seed-{seed}"]
        if mixed_eligibility(context, record, baseline, runs[f"next-item-only/seed-{seed}"], runs[f"strong-kl/seed-{seed}"]) and rollout_eligible(context, rollouts[name], rollouts["baseline"]):
            candidates.append(record)
    return max(candidates, key=lambda row: row["validation"]["top10_proxy_reward"]["hybrid_completion"]) if candidates else None


def decision_record(context: dict, runs: dict, baseline: dict, rollouts: dict) -> dict:
    challenger = choose_mixed(context, runs, baseline, rollouts)
    name = challenger["variant"]["name"] if challenger else None
    confirmation = []
    if name:
        for seed in context["config"]["confirmation_seeds"]:
            key = f"{name}/seed-{seed}"
            if key in runs and f"next-item-only/seed-{seed}" in runs:
                confirmation.append({"training_seed": seed, "eligible": mixed_eligibility(context, runs[key], baseline, runs[f"next-item-only/seed-{seed}"], runs["strong-kl/seed-42"])})
    accepted = name is not None and len(confirmation) == len(context["config"]["confirmation_seeds"]) and all(row["eligible"] for row in confirmation)
    selected = name if accepted else "baseline"
    checkpoint = context["policy_checkpoint"] if not accepted else context["output"] / "runs" / selected / "seed-42" / "policy.pt"
    return {"screening_challenger": name, "confirmation": confirmation, "selected_variant": selected,
            "selected_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "mixed_candidate_accepted": accepted, "holdout_used_for_selection": False}


def holdout_evaluation(context: dict, arrays: dict, observed: dict, cache: dict, logits: dict, decision: dict) -> dict:
    settings = context["config"]["evaluation"]
    rewards = reward_values(cache["raw_reward"], cache["personal_mean"], context["calibration"], context["duration_scale"])
    reference = log_probabilities(logits["baseline"])
    rankings = {name: np.argsort(-value, axis=1, kind="stable")[:, :20] + 3 for name, value in logits.items()}
    rankings.update(control_rankings(arrays, context["catalog"], 20))
    evaluations, vectors = {}, {}
    for name, ranking in rankings.items():
        evaluations[name], vectors[name] = evaluate_policy(arrays, ranking, context["catalog"], rewards, context["counts"], context["known_users"],
                                                           reference, logits=logits.get(name), low_support=settings["low_support_threshold"], observed=observed)
    intervals = {name: mixed_intervals(context, value, vectors["baseline"], arrays["user_ids"]) for name, value in vectors.items() if name != "baseline"}
    selected = decision["selected_variant"]
    secondary, primary = {}, None
    if selected != "baseline":
        bounds = intervals[selected]
        margin = context["config"]["fresh_holdout"]["primary_relative_noninferiority_margin"] * evaluations["baseline"]["ranking"]["metrics"]["ndcg@10"]
        primary = {"comparison": f"{selected} vs baseline", "absolute_ndcg_noninferiority_margin": margin,
                   "ndcg_noninferiority_passed": bounds["ndcg@10"]["ci95"][0] >= -margin,
                   "hybrid_reward_improvement_passed": bounds["hybrid_completion"]["ci95"][0] > 0}
        primary["both_criteria_passed"] = primary["ndcg_noninferiority_passed"] and primary["hybrid_reward_improvement_passed"]
        secondary = {name: mixed_intervals(context, vectors[selected], vectors[name], arrays["user_ids"])
                     for name in context["config"]["fresh_holdout"]["policy_controls"]}
    controls = control_predictions(context["reward_train"], observed)
    raw = cache["raw_reward"][observed["query_indices"], observed["candidate_tokens"] - 3]
    controls.update(transformed(raw, controls["user_mean"], context["calibration"]))
    reward_results = {name: evaluate_reward(observed, value, context["catalog"], context["known_users"],
                                            np.unique(context["reward_train"]["candidate_tokens"]), 10) for name, value in controls.items()}
    return {"policies": evaluations, "intervals_vs_baseline": intervals, "secondary_intervals_for_selected": secondary,
            "primary_confirmation": primary, "reward_model": reward_results, "observed_pairs": describe_rewards(observed), "rankings": rankings}


def load_fresh_queries(context: dict, manifest: dict) -> dict:
    for name, value in manifest["artifacts"].items():
        if sha256(context["holdout_dir"] / name) != value["sha256"]:
            raise ValueError("Fresh holdout artifact changed.")
    with np.load(context["holdout_dir"] / "queries.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    with np.load(context["holdout_dir"] / "users.npz", allow_pickle=False) as saved:
        if np.intersect1d(saved["users"], saved["excluded_users"]).size or not np.all(np.isin(arrays["user_ids"], saved["users"])):
            raise ValueError("Fresh cohort is not disjoint from the original population.")
    if np.intersect1d(arrays["user_ids"], context["known_users"]).size:
        raise ValueError("Fresh users unexpectedly occur in reward training.")
    if not np.all(arrays["history_max_timestamps"] < arrays["timestamps"]):
        raise ValueError("Fresh cohort has a point-in-time violation.")
    return arrays


def attach_fresh_labels(context: dict, arrays: dict) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    try:
        connection.read_parquet(str(context["source"])).create_view("reward_source")
        observed = attach_rewards(connection, arrays, context["catalog"])
    finally:
        connection.close()
    validate_rewards(observed, len(context["catalog"]["items"]))
    return observed


def evaluate_fresh_holdout(context: dict, manifest: dict, decision: dict) -> dict:
    root = context["output"] / "fresh-holdout"
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Incomplete fresh evaluation must be preserved before retrying.")
    root.mkdir(parents=True, exist_ok=True)
    opened = datetime.now(timezone.utc).isoformat()
    print(json.dumps({"stage": "fresh_holdout_opening", "selected": decision["selected_variant"]}), flush=True)
    arrays = load_fresh_queries(context, manifest)
    observed = attach_fresh_labels(context, arrays)
    np.savez_compressed(root / "observed.npz", **observed)
    model = reward_model(context)
    raw = CatalogReward(model, context["known_users"], context["device"]).predict(arrays, context["config"]["execution"]["reward_cache_batch_size"])
    direct = predict_reward(model, observed, context["known_users"], context["device"], context["config"]["execution"]["inference_batch_size"])
    difference = float(np.max(np.abs(direct - raw[observed["query_indices"], observed["candidate_tokens"] - 3])))
    if difference > context["config"]["execution"]["forward_tolerance"]:
        raise ValueError("Fresh reward cache differs from the original forward.")
    personal = control_predictions(context["reward_train"], {**arrays, "candidate_tokens": np.full(len(arrays["lengths"]), 3)})["user_mean"]
    cache = {"raw_reward": raw, "personal_mean": personal}
    np.savez_compressed(root / "reward-cache.npz", **cache)
    names = ["baseline", *context["config"]["fresh_holdout"]["policy_controls"]]
    if decision["selected_variant"] != "baseline":
        names.append(decision["selected_variant"])
    logits, models = {}, {}
    for name in names:
        path = context["policy_checkpoint"] if name == "baseline" else context["output"] / "runs" / name / "seed-42" / "policy.pt"
        policy = new_policy(context)
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        logits[name] = policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        np.savez_compressed(root / f"{name}.npz", logits=logits[name])
        models[name] = {"path": str(path), "sha256": sha256(path)}
        del policy
    evaluation = holdout_evaluation(context, arrays, observed, cache, logits, decision)
    np.savez_compressed(root / "rankings.npz", **evaluation.pop("rankings"))
    verify_mixed_inputs(context)
    result = {"opened_at_utc": opened, "selection_record_sha256": sha256(context["output"] / "selection.json"),
              "holdout_manifest_sha256": sha256(context["holdout_dir"] / "manifest.json"), "models": models,
              "population": {key: manifest[key] for key in ("selected_users", "excluded_original_users", "user_overlap", "queries", "query_users", "targets")},
              "all_users_unknown_to_reward_training": True, "reward_forward_maximum_difference": difference, **evaluation,
              "artifacts": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(root.glob("*.npz"))},
              "selection_changed_on_holdout": False, "historical_test_reevaluated": False}
    write_json(root / "result.json", result)
    return result


def run_mixed(context: dict, caches: dict, manifest: dict) -> dict:
    if (context["output"] / "result.json").exists():
        raise FileExistsError("Mixed series is complete; use --audit to replay it.")
    started = time.perf_counter()
    rewards = {split: reward_values(cache["raw_reward"], cache["personal_mean"], context["calibration"], context["duration_scale"]) for split, cache in caches.items()}
    baseline, vectors = validation_baseline(context, caches, rewards)
    runs = {}
    for variant in context["config"]["variants"]:
        key = f"{variant['name']}/seed-42"
        runs[key] = train_mixed_variant(context, caches, rewards, vectors, variant, 42)
        write_json(context["output"] / "progress.json", {"stage": "screening", "completed_runs": list(runs)})
    rollouts = generate_rollouts(context, runs)
    challenger = choose_mixed(context, runs, baseline, rollouts)
    if challenger:
        sft = next(row for row in context["config"]["variants"] if row["name"] == "next-item-only")
        for seed in context["config"]["confirmation_seeds"]:
            if seed == 42:
                continue
            for variant in (sft, challenger["variant"]):
                key = f"{variant['name']}/seed-{seed}"
                runs[key] = train_mixed_variant(context, caches, rewards, vectors, variant, seed)
    decision = decision_record(context, runs, baseline, rollouts)
    selection_path = context["output"] / "selection.json"
    if selection_path.exists():
        existing = read_json(selection_path)
        if any(existing[name] != value for name, value in decision.items()):
            raise ValueError("A previously frozen selection differs.")
        decision = existing
    else:
        decision.update(recorded_before_holdout_at_utc=datetime.now(timezone.utc).isoformat(), holdout_manifest_sha256=sha256(context["holdout_dir"] / "manifest.json"))
        write_json(selection_path, decision)
    print(json.dumps({"stage": "mixed_selection_frozen", **decision}), flush=True)
    fresh = evaluate_fresh_holdout(context, manifest, decision)
    for record in runs.values():
        audit_mixed_run(context, caches, rewards, vectors, record, False)
    verify_mixed_inputs(context)
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "holdout_preparation": manifest,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(context["device"])},
              "baseline_validation": baseline, "runs": runs, "rollouts": rollouts, "selection": decision, "fresh_holdout": fresh,
              "inprocess_seconds": time.perf_counter() - started, "historical_test_reevaluated": False,
              "reward_model_retrained": False, "fresh_holdout_evaluated": True,
              "limits": ["Adaptive development on the previously used validation, followed by a fixed disjoint-user holdout.",
                         "All new users use the frozen reward model's unknown-user fallback; no adaptation is performed.",
                         "Restricted original catalog and same time period; this is not an independent dataset or online experiment.",
                         "Hybrid reward is a model prediction; causal user benefit and retention are not measured.",
                         "Only one baseline/reward checkpoint; seeds change minibatch order, not the starting model."]}
    write_json(context["output"] / "result.json", result)
    return result


def audit_mixed(context: dict, caches: dict, manifest: dict) -> dict:
    result = read_json(context["output"] / "result.json")
    if result["provenance"] != context["plan"] or result["holdout_preparation"] != manifest:
        raise ValueError("Mixed report provenance differs.")
    rewards = {split: reward_values(cache["raw_reward"], cache["personal_mean"], context["calibration"], context["duration_scale"]) for split, cache in caches.items()}
    baseline, vectors = validation_baseline(context, caches, rewards)
    if baseline != result["baseline_validation"]:
        raise ValueError("Mixed baseline metrics differ.")
    for record in result["runs"].values():
        root = context["output"] / "runs" / record["variant"]["name"] / f"seed-{record['training_seed']}"
        if read_json(root / "summary.json") != record:
            raise ValueError("Mixed saved summary differs.")
        audit_mixed_run(context, caches, rewards, vectors, record, True)
    for modes in result["rollouts"].values():
        for entry in modes.values():
            if sha256(Path(entry["artifact"]["path"])) != entry["artifact"]["sha256"]:
                raise ValueError("Mixed rollout artifact changed.")
    rollouts = generate_rollouts(context, result["runs"])
    decision = decision_record(context, result["runs"], baseline, rollouts)
    if rollouts != result["rollouts"] or any(value != result["selection"][name] for name, value in decision.items()):
        raise ValueError("Mixed rollout or selection does not reproduce.")
    if read_json(context["output"] / "selection.json") != result["selection"]:
        raise ValueError("Frozen mixed selection record changed.")
    fresh, root = result["fresh_holdout"], context["output"] / "fresh-holdout"
    if read_json(root / "result.json") != fresh or sha256(context["output"] / "selection.json") != fresh["selection_record_sha256"]:
        raise ValueError("Fresh evaluation selection binding differs.")
    if result["selection"]["recorded_before_holdout_at_utc"] > fresh["opened_at_utc"]:
        raise ValueError("Fresh holdout was opened before selection was recorded.")
    arrays = load_fresh_queries(context, manifest)
    for name, artifact in fresh["artifacts"].items():
        if sha256(root / name) != artifact["sha256"]:
            raise ValueError("Fresh evaluation artifact changed.")
    with np.load(root / "observed.npz", allow_pickle=False) as saved:
        observed = dict(saved)
    labels = attach_fresh_labels(context, arrays)
    if set(observed) != set(labels) or any(not np.array_equal(value, observed[name]) for name, value in labels.items()):
        raise ValueError("Fresh observed labels do not reproduce.")
    with np.load(root / "reward-cache.npz", allow_pickle=False) as saved:
        cache = dict(saved)
    model = reward_model(context)
    raw = CatalogReward(model, context["known_users"], context["device"]).predict(arrays, context["config"]["execution"]["reward_cache_batch_size"])
    if not np.allclose(raw, cache["raw_reward"], atol=context["config"]["execution"]["forward_tolerance"], rtol=0):
        raise ValueError("Fresh reward forward differs.")
    logits = {}
    for name, entry in fresh["models"].items():
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Fresh evaluated weights changed.")
        with np.load(root / f"{name}.npz", allow_pickle=False) as saved:
            logits[name] = saved["logits"].copy()
        policy = new_policy(context)
        policy.load_state_dict(torch.load(entry["path"], map_location="cpu", weights_only=True), strict=True)
        reproduced = policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        if not np.allclose(reproduced, logits[name], atol=context["config"]["execution"]["forward_tolerance"], rtol=0) or not np.array_equal(np.argsort(-reproduced, axis=1, kind="stable")[:, :20], np.argsort(-logits[name], axis=1, kind="stable")[:, :20]):
            raise ValueError("Fresh policy forward differs.")
        del policy
    evaluation = holdout_evaluation(context, arrays, observed, cache, logits, result["selection"])
    with np.load(root / "rankings.npz", allow_pickle=False) as saved:
        rankings = evaluation.pop("rankings")
        if set(saved.files) != set(rankings) or any(not np.array_equal(value, saved[name]) for name, value in rankings.items()):
            raise ValueError("Fresh saved rankings differ.")
    if any(value != fresh[name] for name, value in evaluation.items()):
        raise ValueError("Fresh metrics or primary confirmation do not reproduce.")
    verify_mixed_inputs(context)
    return {"stage": "mixed_audit_complete", "runs_checked": len(result["runs"]), "rollouts_replayed": 14,
            "selection_recalculated": True, "fresh_population_disjoint": True, "fresh_labels_rejoined": True,
            "all_validation_and_fresh_forwards_checked": True, "all_metrics_recalculated": True,
            "historical_test_reevaluated": False, "selected_variant": result["selection"]["selected_variant"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixed RL/next-item training with a sealed disjoint-user holdout.")
    parser.add_argument("--config", type=Path, default=Path("configs/mixed-objective.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/mixed-objective-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mixed-objective-v1"))
    parser.add_argument("--holdout-dir", type=Path, default=Path("data/processed/mixed-holdout-v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    try:
        context, caches = load_mixed_context(args.config, args.protocol, args.output_dir, args.holdout_dir, args.device, args.resume or args.audit)
        manifest = prepare_fresh_holdout(context)
        if args.audit:
            result = audit_mixed(context, caches, manifest)
        elif args.prepare_only:
            result = {"stage": "mixed_preparation_complete", "holdout_queries": manifest["queries"], "user_overlap": manifest["user_overlap"], "response_labels_read": False}
        else:
            report = run_mixed(context, caches, manifest)
            result = {"stage": "mixed_series_complete", "selected_variant": report["selection"]["selected_variant"],
                      "runs": len(report["runs"]), "primary_confirmation": report["fresh_holdout"]["primary_confirmation"]}
        print(json.dumps(result), flush=True)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
