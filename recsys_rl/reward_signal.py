from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from recsys_rl.data import sha256, write_json
from recsys_rl.model import select_device
from recsys_rl.reward import predict_reward, reward_source_hashes
from recsys_rl.reward_data import load_reward_partition
from recsys_rl.reward_evaluation import control_predictions, evaluate_reward, regression_metrics
from recsys_rl.reward_model import CompletionReward


def fit_calibration(labels: np.ndarray, raw: np.ndarray, personal: np.ndarray) -> dict:
    labels, raw = labels.astype(np.float64), raw.astype(np.float64)
    centered = raw - raw.mean()
    variance = float(np.dot(centered, centered))
    slope = max(0.0, float(np.dot(centered, labels - labels.mean()) / variance)) if variance > 0 else 0.0
    intercept = float(labels.mean() - slope * raw.mean())
    calibrated = np.clip(slope * raw + intercept, 0, 1)
    residual = calibrated - personal
    denominator = float(np.dot(residual, residual))
    alpha = float(np.clip(np.dot(residual, labels - personal) / denominator, 0, 1)) if denominator > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "personal_mix_alpha": alpha, "fit_partition": "selection",
            "fit_pairs": len(labels), "neural_network_retrained": False}


def transformed(raw: np.ndarray, personal: np.ndarray, settings: dict) -> dict:
    calibrated = np.clip(settings["slope"] * raw.astype(np.float64) + settings["intercept"], 0, 1)
    return {"raw": raw, "calibrated": calibrated,
            "hybrid": personal + settings["personal_mix_alpha"] * (calibrated - personal)}


def cluster_interval(values: np.ndarray, users: np.ndarray, repeats: int, seed: int) -> dict | None:
    if not values.size:
        return None
    unique, inverse = np.unique(users, return_inverse=True)
    totals, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, len(unique), size=len(unique))
        estimates[index] = totals[selected].sum() / counts[selected].sum()
    return {"estimate": float(values.mean()), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "clusters": len(unique), "observations": len(values), "bootstrap_repeats": repeats,
            "bootstrap_seed": seed, "scope": "Percentile user-cluster bootstrap on this observed population, with pair weights."}


def candidate_pairs(arrays: dict, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    order = np.argsort(arrays["query_indices"], kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(arrays["query_indices"][order])) + 1, len(order)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:stop]
        if len(indices) < 2:
            continue
        for name in ("user_ids", "timestamps", "lengths"):
            if not np.all(arrays[name][indices] == arrays[name][indices[0]]):
                raise ValueError(f"Candidate comparison has inconsistent {name}.")
        if not np.all(arrays["input_ids"][indices] == arrays["input_ids"][indices[0]]):
            raise ValueError("Candidate comparison has inconsistent histories.")
        if len(np.unique(arrays["candidate_tokens"][indices])) != len(indices):
            raise ValueError("Duplicate candidate in a query.")
        for position, first in enumerate(indices):
            for second in indices[position + 1:]:
                if abs(float(arrays["rewards"][first]) - float(arrays["rewards"][second])) > tolerance:
                    left.append(first)
                    right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def compare_candidates(arrays: dict, predictions: dict, catalog: dict, config: dict) -> dict:
    left, right = candidate_pairs(arrays, config["label_tolerance"])
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else 0.0 for item in catalog["items"]])
    predictions = {**predictions, "shorter_first": -durations[arrays["candidate_tokens"] - 3]}
    outcome = arrays["rewards"][left] - arrays["rewards"][right]
    result = {}
    for group, mask in {"unequal_labels": np.ones(len(left), dtype=bool),
                        "gap_at_least_0_10": np.abs(outcome) >= config["strong_label_gap"] - config["label_tolerance"]}.items():
        first, second = left[mask], right[mask]
        scores = {}
        for name, values in predictions.items():
            difference = values[first] - values[second]
            scores[name] = np.where(np.abs(difference) <= config["prediction_tolerance"], .5,
                                    (np.sign(difference) == np.sign(outcome[mask])).astype(np.float64))
        users = arrays["user_ids"][first]
        queries, inverse = np.unique(arrays["query_indices"][first], return_inverse=True)
        result[group] = {"comparisons": len(first), "queries": len(queries), "users": len(np.unique(users)),
                         "accuracy": {name: cluster_interval(value, users, config["bootstrap_repeats"], config["bootstrap_seed"])
                                      for name, value in scores.items()},
                         "query_macro_accuracy": {name: float(np.mean(np.bincount(inverse, weights=value) / np.bincount(inverse)))
                                                  if value.size else None for name, value in scores.items()},
                         "raw_minus_item_mean": cluster_interval(scores["raw"] - scores["item_mean"], users,
                                                                  config["bootstrap_repeats"], config["bootstrap_seed"])}
    return result


def permuted_candidates(arrays: dict, catalog: dict, seed: int, within_duration: bool) -> np.ndarray:
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else 0.0 for item in catalog["items"]])
    known = durations[durations > 0]
    cuts = np.quantile(known, [1 / 3, 2 / 3]) if known.size else np.asarray([0, 0])
    classes = np.searchsorted(cuts, durations[arrays["candidate_tokens"] - 3], side="left") if within_duration else np.zeros(len(arrays["rewards"]), dtype=np.int64)
    classes[durations[arrays["candidate_tokens"] - 3] <= 0] = 3 if within_duration else 0
    keys = np.column_stack((arrays["user_ids"], classes))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    result = arrays["candidate_tokens"].copy()
    for group in range(int(inverse.max()) + 1):
        positions = np.flatnonzero(inverse == group)
        result[positions] = arrays["candidate_tokens"][rng.permutation(positions)]
    return result


def run_reward_signal(data_dir: Path, output_dir: Path, config_path: Path, protocol_path: Path, requested_device: str) -> dict:
    config = json.loads(config_path.read_text())
    if len(config["permutation_seeds"]) != 3 or config["bootstrap_repeats"] < 100:
        raise ValueError("Signal protocol requires three permutations and at least 100 bootstrap repeats.")
    report_path, reward_dir = config_path.parent / config["reward_report"], config_path.parent / config["reward_output_dir"]
    report = json.loads(report_path.read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if sha256(data_dir / "manifest.json") != report["preparation_manifest_sha256"]:
        raise ValueError("Reward preparation differs from the trained model.")
    protected = {name: value for name, value in report["source_files_sha256"].items() if name != "cli.py"}
    for name, expected in protected.items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Original reward computation changed: {name}")
    for name, values in report["artifacts"].items():
        if sha256(reward_dir / name) != values["sha256"]:
            raise ValueError(f"Original reward artifact changed: {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Signal output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"config": sha256(config_path), "protocol": sha256(protocol_path), "reference_report": sha256(report_path),
              "preparation": sha256(data_dir / "manifest.json"), "source_files": reward_source_hashes()}
    torch.set_num_threads(4)
    device = select_device(requested_device)
    catalog_path = Path(manifest["policy_data_dir"]) / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    if sha256(catalog_path) != manifest["policy_catalog_sha256"]:
        raise ValueError("Original catalog changed.")
    checkpoint = torch.load(reward_dir / "reward.pt", map_location="cpu", weights_only=True)
    known_users = checkpoint["state_dict"]["known_users"].numpy().astype(np.uint32)
    model = CompletionReward(catalog, manifest["history_length"], known_users, report["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train = load_reward_partition(data_dir, "train")
    known_items = np.unique(train["candidate_tokens"])
    arrays_by_split, raw_by_split = {}, {}
    maximum_difference = 0.0
    for split in ("selection", "validation"):
        arrays = load_reward_partition(data_dir, split)
        with np.load(reward_dir / f"{split}-predictions.npz", allow_pickle=False) as saved:
            for name in ("user_ids", "timestamps", "candidate_tokens", "query_indices", "rewards"):
                if not np.array_equal(saved[name], arrays[name]):
                    raise ValueError(f"Original signal population differs: {split}/{name}")
            raw = saved["model"].copy()
        reproduced = predict_reward(model, arrays, known_users, device, report["config"]["training"]["batch_size"])
        difference = float(np.max(np.abs(reproduced - raw)))
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-6:
            raise ValueError("Loaded checkpoint does not reproduce saved reward predictions.")
        arrays_by_split[split], raw_by_split[split] = arrays, raw
    selection = arrays_by_split["selection"]
    calibrator = fit_calibration(selection["rewards"], raw_by_split["selection"], control_predictions(train, selection)["user_mean"])
    write_json(output_dir / "calibration.json", calibrator)
    results = {}
    started = time.perf_counter()
    for split, arrays in arrays_by_split.items():
        controls = control_predictions(train, arrays)
        predictions = {**transformed(raw_by_split[split], controls["user_mean"], calibrator), **controls}
        payload = {name: arrays[name] for name in ("user_ids", "timestamps", "candidate_tokens", "query_indices", "rewards")}
        payload.update(predictions)
        metrics = {name: evaluate_reward(arrays, values, catalog, known_users, known_items, config["calibration_bins"])
                   for name, values in predictions.items()}
        personal_errors = (controls["user_mean"] - arrays["rewards"].astype(np.float64)) ** 2
        intervals = {name: cluster_interval((values.astype(np.float64) - arrays["rewards"]) ** 2 - personal_errors,
                                            arrays["user_ids"], config["bootstrap_repeats"], config["bootstrap_seed"])
                     for name, values in predictions.items() if name in ("raw", "calibrated", "hybrid")}
        pairwise = compare_candidates(arrays, predictions, catalog, config)
        permutations = {}
        for group, within_duration in (("user", False), ("user_duration", True)):
            permutations[group] = []
            for seed in config["permutation_seeds"]:
                candidates = permuted_candidates(arrays, catalog, seed, within_duration)
                altered = {**arrays, "candidate_tokens": candidates}
                raw = predict_reward(model, altered, known_users, device, report["config"]["training"]["batch_size"])
                changed = candidates != arrays["candidate_tokens"]
                key = f"perm_{group}_{seed}"
                payload[f"{key}_candidates"], payload[f"{key}_raw"] = candidates, raw
                permutations[group].append({"seed": seed, "changed_pairs": int(changed.sum()),
                    "changed_fraction": float(changed.mean()), "raw_metrics": regression_metrics(arrays["rewards"], raw),
                    "mse_change": regression_metrics(arrays["rewards"], raw)["mse"] - metrics["raw"]["metrics"]["mse"],
                    "original_changed_pairs": regression_metrics(arrays["rewards"][changed], raw_by_split[split][changed]),
                    "permuted_changed_pairs": regression_metrics(arrays["rewards"][changed], raw[changed])})
        path = output_dir / f"{split}-predictions.npz"
        np.savez_compressed(path, **payload)
        with np.load(path, allow_pickle=False) as saved:
            for name, values in predictions.items():
                if regression_metrics(arrays["rewards"], saved[name]) != metrics[name]["metrics"]:
                    raise ValueError("Signal metrics differ after prediction serialization.")
            for group, rows in permutations.items():
                for row in rows:
                    key = f"perm_{group}_{row['seed']}"
                    if not np.array_equal(saved[f"{key}_candidates"], permuted_candidates(arrays, catalog, row["seed"], group == "user_duration")):
                        raise ValueError("Saved candidate permutation differs.")
                    if regression_metrics(arrays["rewards"], saved[f"{key}_raw"]) != row["raw_metrics"]:
                        raise ValueError("Saved permutation metrics differ.")
        results[split] = {"evaluation": metrics, "mse_difference_vs_personal": intervals,
                          "same_query_candidate_accuracy": pairwise, "candidate_permutations": permutations}
        print(json.dumps({"stage": "reward_signal_split_completed", "split": split,
                          "mse": {name: value["metrics"]["mse"] for name, value in metrics.items()}}), flush=True)
    if reward_source_hashes() != frozen["source_files"] or sha256(config_path) != frozen["config"] or sha256(protocol_path) != frozen["protocol"] or sha256(report_path) != frozen["reference_report"] or sha256(data_dir / "manifest.json") != frozen["preparation"]:
        raise ValueError("Signal experiment inputs changed during execution.")
    parent = Path(manifest["policy_data_dir"])
    for split, expected in manifest["policy_partitions_sha256"].items():
        if sha256(parent / f"{split}.npz") != expected:
            raise ValueError(f"Original policy data changed: {split}")
    if sha256(catalog_path) != manifest["policy_catalog_sha256"] or sha256(Path(manifest["source_path"])) != manifest["source"]["sha256"]:
        raise ValueError("Original catalog or raw data changed.")
    policy_checkpoint = manifest["reference_policy_checkpoint"]
    if sha256(Path(policy_checkpoint["path"])) != policy_checkpoint["sha256"]:
        raise ValueError("Original policy checkpoint changed.")
    for name, values in report["artifacts"].items():
        if sha256(reward_dir / name) != values["sha256"]:
            raise ValueError(f"Original reward artifact changed: {name}")
    result = {"experiment": config["experiment"], "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "config": config, "frozen": frozen, "calibration": calibrator, "device": str(device),
              "maximum_original_prediction_difference": maximum_difference, **results,
              "analysis_seconds": time.perf_counter() - started,
              "artifacts": {name: {"sha256": sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
                            for name in ("calibration.json", "selection-predictions.npz", "validation-predictions.npz")},
              "audit": {"saved_metrics_recalculated": True, "permutations_reproduced": True,
                        "original_reward_artifacts_unchanged": True, "policy_data_unchanged": True},
              "limits": ["Adaptive continuation on an already inspected validation; final test is held out.",
                         "Calibration and personal mixing can improve MSE without improving candidate ordering.",
                         "Same-query comparisons are observational, conditional on multiple listened items and rounded timestamps.",
                         "Permuted candidate inputs retain the original item's label; this is an input ablation, not a counterfactual response estimate.",
                         "User-cluster bootstrap does not remove adaptive selection bias or uncertainty from training a single reward model."],
              "reward_model_retrained": False, "test_evaluated": False, "rl_trained": False}
    write_json(output_dir / "result.json", result)
    return result
