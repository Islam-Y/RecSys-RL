from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch

from recsys_rl.data import load_config, load_partition, sha256, write_json
from recsys_rl.evaluation import control_rankings
from recsys_rl.model import select_device
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import evaluate_policy, log_probabilities, paired_policy_intervals, reward_values
from recsys_rl.reward import predict_reward, reward_source_hashes
from recsys_rl.reward_catalog import CatalogReward
from recsys_rl.reward_data import attach_rewards, describe_rewards, load_reward_partition, validate_rewards
from recsys_rl.reward_evaluation import control_predictions, evaluate_reward
from recsys_rl.reward_model import CompletionReward
from recsys_rl.reward_signal import compare_candidates, transformed


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(context: dict) -> None:
    if reward_source_hashes() != context["frozen"]["source_files"]:
        raise ValueError("Final evaluation source changed.")
    for path, expected in context["frozen"]["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Final evaluation input changed: {path}")


def prepare_context(config_path: Path, protocol_path: Path, output_dir: Path, requested_device: str, audit: bool) -> dict:
    config_path, protocol_path, output_dir = (path.resolve() for path in (config_path, protocol_path, output_dir))
    config = read_json(config_path)
    if config["training_allowed"] or config["selection_on_test_allowed"]:
        raise ValueError("Final evaluation cannot train or select on test.")
    if config["evaluation"]["top_k"] != [5, 10, 20] or config["controls"] != ["train_popularity", "recent_items"]:
        raise ValueError("Unsupported final metric cutoffs or controls.")
    if [row["name"] for row in config["policies"]] != ["baseline", "raw-kl", "raw-strong-kl"]:
        raise ValueError("Final model list differs from the prespecified comparisons.")
    pilot_path = (config_path.parent / config["pilot_report"]).resolve()
    execution_path = (config_path.parent / config["pilot_execution"]).resolve()
    pilot, execution = read_json(pilot_path), read_json(execution_path)
    if sha256(pilot_path) != execution["result_sha256"] or pilot["test_evaluated"]:
        raise ValueError("Pilot report or held-out status differs.")
    selected = [row for row in config["policies"] if row["role"] == "selected_primary"]
    if len(selected) != 1 or selected[0]["name"] != pilot["selected_variant"] or selected[0]["training_seed"] != 42:
        raise ValueError("Final primary model differs from the validation selection.")
    for name, expected in pilot["provenance"]["frozen"]["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical pilot computation changed: {name}")
    pilot_config_path = Path(pilot["paths"]["config_path"])
    pilot_config = read_json(pilot_config_path)
    report_paths = {name: (pilot_config_path.parent / pilot_config[f"{name}_report"]).resolve()
                    for name in ("policy", "reward", "signal")}
    reports = {name: read_json(path) for name, path in report_paths.items()}
    data_dir, reward_data_dir = Path(pilot["paths"]["data_dir"]), Path(pilot["paths"]["reward_data_dir"])
    reward_dir = (pilot_config_path.parent / pilot_config["reward_output_dir"]).resolve()
    policy_config_path = (pilot_config_path.parent / pilot_config["policy_config"]).resolve()
    signal_config_path = (config_path.parent / config["signal_config"]).resolve()
    models = {}
    for row in config["policies"]:
        if row["name"] == "baseline":
            checkpoint = reports["policy"]["selected_checkpoint"]
            path = Path(checkpoint["path"]).resolve()
        else:
            record = pilot["runs"][f"{row['name']}/seed-{row['training_seed']}"]
            checkpoint = record["artifacts"]["policy.pt"]
            path = Path(pilot["selected_checkpoint"]["path"]).parents[2] / row["name"] / f"seed-{row['training_seed']}" / "policy.pt"
        models[row["name"]] = {**row, "path": str(path), "sha256": checkpoint["sha256"]}
    inputs = {str(Path(path).resolve()): expected for path, expected in pilot["provenance"]["frozen"]["inputs"].items()}
    for path in (config_path, protocol_path, pilot_path, execution_path, signal_config_path):
        inputs[str(path)] = sha256(path)
    inputs.update({row["path"]: row["sha256"] for row in models.values()})
    frozen = {"inputs": inputs, "source_files": reward_source_hashes()}
    context = {"config": config, "frozen": frozen, "output": output_dir, "models": models,
               "data_dir": data_dir, "reward_data_dir": reward_data_dir, "reward_dir": reward_dir,
               "policy_config": load_config(policy_config_path), "policy_checkpoint": Path(models["baseline"]["path"]),
               "reward_report": reports["reward"], "calibration": reports["signal"]["calibration"],
               "signal_config": read_json(signal_config_path), "device": select_device(requested_device),
               "manifest": read_json(data_dir / "manifest.json"), "catalog": read_json(data_dir / "catalog.json"),
               "source": Path(read_json(reward_data_dir / "manifest.json")["source_path"]).resolve(),
               "paths": {"config": str(config_path), "protocol": str(protocol_path)},
               "pilot_source_commit": pilot["provenance"]["source_commit"]}
    verify_inputs(context)
    plan_path = output_dir / "plan.json"
    if audit:
        context["plan"] = read_json(plan_path)
        if context["plan"]["frozen"] != frozen or context["plan"]["models"] != models:
            raise ValueError("Final audit differs from the frozen plan.")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Final result already exists; use --audit or a separate output directory: {output_dir}")
        context["plan"] = {"config": config, "frozen": frozen, "models": models,
                           "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                           "recorded_before_test_load_utc": datetime.now(timezone.utc).isoformat(),
                           "selection_frozen_before_test": True}
        write_json(plan_path, context["plan"])
    torch.set_num_threads(4)
    train = load_reward_partition(reward_data_dir, "train")
    context["reward_train"] = train
    context["known_users"] = np.unique(train["user_ids"])
    context["counts"] = np.bincount(train["candidate_tokens"] - 3, minlength=len(context["catalog"]["items"]))
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan
                            for item in context["catalog"]["items"]])
    known = durations[np.isfinite(durations) & (durations > 0)]
    cap = float(np.quantile(known, pilot_config["reward"]["duration_cap_catalog_quantile"]))
    if cap != pilot["cache"]["duration_cap_seconds"]:
        raise ValueError("Frozen duration cap differs.")
    context["duration_scale"] = np.minimum(np.where(np.isfinite(durations) & (durations > 0), durations, np.median(known)), cap) / cap
    return context


def load_test(context: dict) -> dict:
    arrays = load_partition(context["data_dir"], "test")
    cutoffs = context["manifest"]["cutoffs_seconds"]
    if np.any((arrays["timestamps"] <= cutoffs["validation_end"]) | (arrays["timestamps"] > cutoffs["data_end"])):
        raise ValueError("Final test temporal bounds differ.")
    if len(arrays["lengths"]) != context["manifest"]["partitions"]["test"]["queries"]:
        raise ValueError("Final test population differs.")
    return arrays


def reward_model(context: dict) -> CompletionReward:
    model = CompletionReward(context["catalog"], context["policy_config"]["data"]["history_length"],
                             context["known_users"], context["reward_report"]["config"]["model"]).to(context["device"])
    checkpoint = torch.load(context["reward_dir"] / "reward.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluate_saved(context: dict, arrays: dict, observed: dict, cache: dict, logits: dict) -> dict:
    settings, train = context["config"]["evaluation"], context["reward_train"]
    rewards = reward_values(cache["raw_reward"], cache["personal_mean"], context["calibration"], context["duration_scale"])
    rankings = {name: np.argsort(-value, axis=1, kind="stable")[:, :20] + 3 for name, value in logits.items()}
    rankings.update(control_rankings(arrays, context["catalog"], 20))
    reference_logp = log_probabilities(logits["baseline"])
    evaluations, vectors = {}, {}
    for name, ranking in rankings.items():
        evaluations[name], vectors[name] = evaluate_policy(arrays, ranking, context["catalog"], rewards, context["counts"],
                                                           context["known_users"], reference_logp, logits=logits.get(name),
                                                           low_support=settings["low_support_threshold"], observed=observed)
    intervals = {}
    for name in evaluations:
        if name == "baseline":
            continue
        intervals[name] = paired_policy_intervals(vectors[name], vectors["baseline"], arrays["user_ids"],
                                                  settings["bootstrap_repeats"], settings["bootstrap_seed"])
        for value in intervals[name].values():
            value["scope"] = "Percentile user-cluster bootstrap on test queries, with equal query weights."
    predicted = control_predictions(train, observed)
    raw = cache["raw_reward"][observed["query_indices"], observed["candidate_tokens"] - 3]
    predicted.update(transformed(raw, predicted["user_mean"], context["calibration"]))
    reward_results = {name: evaluate_reward(observed, values, context["catalog"], context["known_users"],
                                            np.unique(train["candidate_tokens"]), settings["calibration_bins"])
                      for name, values in predicted.items()}
    return {"policies": evaluations, "paired_intervals_vs_baseline": intervals, "reward_model": reward_results,
            "candidate_comparison": compare_candidates(observed, predicted, context["catalog"], context["signal_config"]),
            "observed_test_pairs": describe_rewards(observed), "rankings": rankings}


def run_final(context: dict) -> dict:
    started = time.perf_counter()
    print(json.dumps({"stage": "final_test_opening", "models": context["models"]}), flush=True)
    arrays = load_test(context)
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    try:
        connection.read_parquet(str(context["source"])).create_view("reward_source")
        observed = attach_rewards(connection, arrays, context["catalog"])
    finally:
        connection.close()
    validate_rewards(observed, len(context["catalog"]["items"]))
    root, settings = context["output"], context["config"]["evaluation"]
    np.savez_compressed(root / "observed-test.npz", **observed)
    model = reward_model(context)
    raw = CatalogReward(model, context["known_users"], context["device"]).predict(arrays, settings["reward_cache_batch_size"])
    direct = predict_reward(model, observed, context["known_users"], context["device"], settings["inference_batch_size"])
    difference = float(np.max(np.abs(direct - raw[observed["query_indices"], observed["candidate_tokens"] - 3])))
    if difference > settings["reward_forward_tolerance"]:
        raise ValueError("Final catalog reward differs from the original forward.")
    personal = control_predictions(context["reward_train"], {**arrays, "candidate_tokens": np.full(len(arrays["lengths"]), 3)})["user_mean"]
    cache = {"raw_reward": raw, "personal_mean": personal}
    np.savez_compressed(root / "reward-cache.npz", **cache)
    logits = {}
    for name, entry in context["models"].items():
        policy = new_policy(context)
        policy.load_state_dict(torch.load(entry["path"], map_location="cpu", weights_only=True), strict=True)
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        logits[name] = policy_logits(policy, arrays, context["device"], settings["inference_batch_size"])
        if not np.all(np.isfinite(logits[name])):
            raise ValueError("Non-finite final policy logits.")
        np.savez_compressed(root / f"{name}-test.npz", logits=logits[name],
                            **{key: arrays[key] for key in ("user_ids", "timestamps", "target_tokens", "target_offsets")})
        print(json.dumps({"stage": "final_policy_forward_complete", "policy": name}), flush=True)
        del policy
    evaluation = evaluate_saved(context, arrays, observed, cache, logits)
    np.savez_compressed(root / "rankings.npz", **evaluation.pop("rankings"))
    verify_inputs(context)
    filenames = ["observed-test.npz", "reward-cache.npz", "rankings.npz", *[f"{name}-test.npz" for name in logits]]
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "paths": context["paths"],
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(context["device"])},
              **evaluation, "test_queries": len(arrays["lengths"]), "test_users": len(np.unique(arrays["user_ids"])),
              "test_targets": len(arrays["target_tokens"]), "excluded_missing_reward_labels": len(arrays["target_tokens"]) - len(observed["rewards"]),
              "reward_forward_maximum_difference": difference, "calibration": context["calibration"],
              "primary_comparison": "raw-kl vs baseline", "secondary_comparison": "raw-strong-kl vs baseline",
              "artifacts": {name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size} for name in filenames},
              "inprocess_seconds": time.perf_counter() - started, "training_performed": False,
              "selection_changed_on_test": False, "test_evaluated": True,
              "limits": ["Fixed temporal test on the same selected user population and restricted 2000-item catalog.",
                         "Reward is a prediction conditional on observed listens; off-distribution responses are unknown.",
                         "Observed matched outcomes are selected, not unbiased policy value estimates.",
                         "Secondary comparisons and intervals are descriptive; no multiple-comparison correction.",
                         "The trained objective is one-step; multi-step results come from the earlier synthetic validation rollout.",
                         "No causal satisfaction or retention improvement is measured."]}
    write_json(root / "result.json", result)
    return result


def audit_final(context: dict) -> dict:
    root = context["output"]
    result = read_json(root / "result.json")
    if result["provenance"] != context["plan"] or result["training_performed"] or result["selection_changed_on_test"]:
        raise ValueError("Final result provenance or phase flags differ.")
    for name, entry in result["artifacts"].items():
        if sha256(root / name) != entry["sha256"]:
            raise ValueError(f"Final artifact changed: {name}")
    arrays = load_test(context)
    with np.load(root / "observed-test.npz", allow_pickle=False) as saved:
        observed = dict(saved)
    validate_rewards(observed, len(context["catalog"]["items"]))
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    try:
        connection.read_parquet(str(context["source"])).create_view("reward_source")
        reproduced = attach_rewards(connection, arrays, context["catalog"])
    finally:
        connection.close()
    if set(reproduced) != set(observed) or any(not np.array_equal(value, observed[name]) for name, value in reproduced.items()):
        raise ValueError("Observed test labels do not reproduce from the frozen raw source.")
    with np.load(root / "reward-cache.npz", allow_pickle=False) as saved:
        cache = dict(saved)
    personal = control_predictions(context["reward_train"], {**arrays, "candidate_tokens": np.full(len(arrays["lengths"]), 3)})["user_mean"]
    if not np.array_equal(personal, cache["personal_mean"]):
        raise ValueError("Frozen personal reward means do not reproduce.")
    model = reward_model(context)
    reproduced_raw = CatalogReward(model, context["known_users"], context["device"]).predict(arrays, context["config"]["evaluation"]["reward_cache_batch_size"])
    reward_difference = float(np.max(np.abs(reproduced_raw - cache["raw_reward"])))
    if reward_difference > context["config"]["evaluation"]["reward_forward_tolerance"]:
        raise ValueError("Full test catalog reward does not reproduce.")
    logits, differences = {}, {}
    for name, entry in context["models"].items():
        with np.load(root / f"{name}-test.npz", allow_pickle=False) as saved:
            for field in ("user_ids", "timestamps", "target_tokens", "target_offsets"):
                if not np.array_equal(saved[field], arrays[field]):
                    raise ValueError(f"Final policy test population differs: {name}/{field}")
            logits[name] = saved["logits"].copy()
        policy = new_policy(context)
        policy.load_state_dict(torch.load(entry["path"], map_location="cpu", weights_only=True), strict=True)
        reproduced = policy_logits(policy, arrays, context["device"], context["config"]["evaluation"]["inference_batch_size"])
        differences[name] = float(np.max(np.abs(reproduced - logits[name])))
        if differences[name] > context["config"]["evaluation"]["policy_forward_tolerance"] or not np.array_equal(np.argsort(-reproduced, axis=1, kind="stable")[:, :20], np.argsort(-logits[name], axis=1, kind="stable")[:, :20]):
            raise ValueError(f"Final policy forward or rankings do not reproduce: {name}")
        del policy
    evaluation = evaluate_saved(context, arrays, observed, cache, logits)
    with np.load(root / "rankings.npz", allow_pickle=False) as saved:
        rankings = evaluation.pop("rankings")
        if set(saved.files) != set(rankings) or any(not np.array_equal(value, saved[name]) for name, value in rankings.items()):
            raise ValueError("Final saved rankings differ.")
    if any(result[name] != value for name, value in evaluation.items()):
        raise ValueError("Final metrics, intervals, or reward evaluation do not reproduce.")
    verify_inputs(context)
    return {"stage": "final_evaluation_audit_complete", "policy_forward_maximum_differences": differences,
            "full_catalog_reward_maximum_difference": reward_difference, "observed_labels_rejoined": True,
            "rankings_metrics_intervals_and_controls_recalculated": True, "frozen_inputs_unchanged": True,
            "training_performed": False, "selection_changed_on_test": False, "test_evaluated": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen models once on temporal test, or replay its saved artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/final-evaluation.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/final-evaluation-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final-evaluation-v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    try:
        context = prepare_context(args.config, args.protocol, args.output_dir, args.device, args.audit)
        if args.audit:
            result = audit_final(context)
        else:
            result = run_final(context)
            result = {"stage": "final_evaluation_complete", "policies": {name: row["ranking"]["metrics"] for name, row in result["policies"].items()}, "test_evaluated": True}
        print(json.dumps(result), flush=True)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
