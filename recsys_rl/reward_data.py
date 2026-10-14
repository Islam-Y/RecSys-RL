from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from recsys_rl.data import FIRST_ITEM, load_partition, sha256, write_json


def load_reward_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or not 0 <= config["seed"] < 2 ** 32:
        raise ValueError("Invalid reward training seed.")
    if not 0 < config["data"]["selection_quantile"] < 1:
        raise ValueError("Reward selection quantile must be in (0, 1).")
    if config["data"]["label"] != "mean_of_clipped_played_ratio" or config["data"]["missing_label"] != "exclude_pair":
        raise ValueError("Unsupported reward label rule.")
    model, training = config["model"], config["training"]
    if min(model["embedding_size"], model["user_embedding_size"], *model["hidden_sizes"]) < 1 or len(model["hidden_sizes"]) != 2:
        raise ValueError("Reward embedding and hidden dimensions must be positive; two hidden layers are required.")
    if not 0 <= model["dropout"] < 1:
        raise ValueError("Reward dropout must be in [0, 1).")
    if min(training["epochs"], training["early_stopping_patience"], training["batch_size"], training["learning_rate"], training["gradient_clip"]) <= 0:
        raise ValueError("Reward training budgets must be positive.")
    if training["weight_decay"] < 0 or training["checkpoint_metric"] != "mse":
        raise ValueError("Reward selection uses MSE and nonnegative weight decay.")
    if config["evaluation"]["calibration_bins"] < 2:
        raise ValueError("At least two calibration bins are required.")
    return config


def attach_rewards(connection, queries: dict, catalog: dict) -> dict:
    query_indices = np.repeat(np.arange(len(queries["lengths"]), dtype=np.int32), np.diff(queries["target_offsets"]))
    items = np.asarray([item["item_id"] for item in catalog["items"]], dtype=np.uint32)
    candidates = queries["target_tokens"]
    requests = {"pair_index": np.arange(len(candidates), dtype=np.int32),
                "user_id": queries["user_ids"][query_indices], "target_time": queries["timestamps"][query_indices],
                "item_id": items[candidates - FIRST_ITEM]}
    connection.register("reward_requests", requests)
    try:
        rows = connection.execute("""
            SELECT pair_index, count(e.uid), count(e.played_ratio_pct),
                   avg(CASE WHEN played_ratio_pct IS NOT NULL THEN least(played_ratio_pct, 100) / 100.0 END),
                   min(played_ratio_pct), max(played_ratio_pct),
                   count(*) FILTER (WHERE played_ratio_pct > 100),
                   count(*) FILTER (WHERE played_ratio_pct = 0), min(is_organic), max(is_organic)
            FROM reward_requests r LEFT JOIN reward_source e
              ON e.uid = r.user_id AND e.timestamp = r.target_time AND e.item_id = r.item_id AND e.event_type = 'listen'
            GROUP BY pair_index ORDER BY pair_index
        """).fetchall()
    finally:
        connection.unregister("reward_requests")
    if len(rows) != len(candidates) or any(row[0] != index for index, row in enumerate(rows)):
        raise ValueError("Reward join changed the requested population.")
    if any(row[1] == 0 for row in rows):
        raise ValueError("A prepared next-item target is absent from the raw listen events.")
    valid = np.asarray([row[2] > 0 for row in rows])
    if not valid.any():
        raise ValueError("No observed playback labels.")
    kept = [row for row in rows if row[2] > 0]
    selected_queries = query_indices[valid]
    organic = [row[8] if row[8] == row[9] and row[8] in (0, 1) else 2 for row in kept]
    return {"input_ids": queries["input_ids"][selected_queries], "lengths": queries["lengths"][selected_queries],
            "candidate_tokens": candidates[valid], "query_indices": selected_queries,
            "user_ids": queries["user_ids"][selected_queries], "timestamps": queries["timestamps"][selected_queries],
            "history_max_timestamps": queries["history_max_timestamps"][selected_queries],
            "rewards": np.asarray([row[3] for row in kept], dtype=np.float32),
            "observed_events": np.asarray([row[1] for row in kept], dtype=np.int32),
            "label_events": np.asarray([row[2] for row in kept], dtype=np.int32),
            "raw_ratio_min": np.asarray([row[4] for row in kept], dtype=np.uint16),
            "raw_ratio_max": np.asarray([row[5] for row in kept], dtype=np.uint16),
            "events_above_100": np.asarray([row[6] for row in kept], dtype=np.int32),
            "zero_events": np.asarray([row[7] for row in kept], dtype=np.int32),
            "organic": np.asarray(organic, dtype=np.int8)}


def describe_rewards(arrays: dict) -> dict:
    rewards = arrays["rewards"]
    return {"pairs": len(rewards), "queries": len(np.unique(arrays["query_indices"])),
            "users": len(np.unique(arrays["user_ids"])), "items": len(np.unique(arrays["candidate_tokens"])),
            "timestamp_min": int(arrays["timestamps"].min()), "timestamp_max": int(arrays["timestamps"].max()),
            "observed_events": int(arrays["observed_events"].sum()), "label_events": int(arrays["label_events"].sum()),
            "pairs_with_duplicate_events": int(np.count_nonzero(arrays["observed_events"] > 1)),
            "events_above_100": int(arrays["events_above_100"].sum()), "zero_events": int(arrays["zero_events"].sum()),
            "raw_ratio_min": int(arrays["raw_ratio_min"].min()), "raw_ratio_max": int(arrays["raw_ratio_max"].max()),
            "reward_mean": float(rewards.astype(np.float64).mean()), "reward_std": float(rewards.astype(np.float64).std()),
            "reward_quantiles": np.quantile(rewards, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            "full_completion_fraction": float(np.mean(rewards == 1)),
            "organic_pairs": {str(value): int(np.count_nonzero(arrays["organic"] == value)) for value in (0, 1, 2)}}


def prepare_reward(source: Path, data_dir: Path, output_dir: Path, config_path: Path, protocol_path: Path) -> dict:
    config = load_reward_config(config_path)
    parent_path = data_dir / "manifest.json"
    parent = json.loads(parent_path.read_text())
    parent_hash = sha256(parent_path)
    if sha256(source) != parent["source"]["sha256"]:
        raise ValueError("Reward source differs from the original policy data.")
    if sha256(data_dir / "catalog.json") != parent["catalog_sha256"]:
        raise ValueError("Policy catalog changed.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Reward preparation output is not empty: {output_dir}")
    reference_path = config_path.parent / config["reference_policy_report"]
    reference = json.loads(reference_path.read_text())
    if reference["provenance"]["frozen"]["preparation_manifest_sha256"] != parent_hash:
        raise ValueError("Selected policy used different prepared data.")
    catalog = json.loads((data_dir / "catalog.json").read_text())
    train_queries = load_partition(data_dir, "train")
    validation_queries = load_partition(data_dir, "validation")
    cutoff = int(np.quantile(train_queries["timestamps"], config["data"]["selection_quantile"], method="inverted_cdf"))
    if not int(train_queries["timestamps"].min()) < cutoff < parent["cutoffs_seconds"]["train_end"]:
        raise ValueError("Cannot form a temporal reward selection period.")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    try:
        connection.read_parquet(str(source)).create_view("reward_source")
        train_pairs = attach_rewards(connection, train_queries, catalog)
        validation = attach_rewards(connection, validation_queries, catalog)
    finally:
        connection.close()
    mask = train_pairs["timestamps"] <= cutoff
    partitions = {"train": {name: values[mask] for name, values in train_pairs.items()},
                  "selection": {name: values[~mask] for name, values in train_pairs.items()}, "validation": validation}
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, arrays in partitions.items():
        validate_rewards(arrays, len(catalog["items"]))
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        summaries[split] = {**describe_rewards(arrays), "sha256": sha256(path)}
        print(json.dumps({"stage": "reward_partition_prepared", "split": split, "pairs": len(arrays["rewards"])}), flush=True)
    for split, values in parent["partitions"].items():
        if sha256(data_dir / f"{split}.npz") != values["sha256"]:
            raise ValueError(f"Original policy partition changed: {split}")
    if sha256(parent_path) != parent_hash or sha256(data_dir / "catalog.json") != parent["catalog_sha256"]:
        raise ValueError("Policy preparation changed during reward preparation.")
    result = {"experiment": config["experiment"], "config": config, "config_sha256": sha256(config_path),
              "protocol_sha256": sha256(protocol_path), "policy_data_dir": str(data_dir),
              "policy_manifest_sha256": parent_hash, "policy_catalog_sha256": parent["catalog_sha256"],
              "policy_partitions_sha256": {name: value["sha256"] for name, value in parent["partitions"].items()},
              "reference_policy_report_sha256": sha256(reference_path), "reference_policy_checkpoint": reference["selected_checkpoint"],
              "source": parent["source"], "source_path": str(source), "catalog_size": len(catalog["items"]),
              "history_length": parent["config"]["data"]["history_length"],
              "cutoffs_seconds": {"reward_train_end": cutoff, "selection_end": parent["cutoffs_seconds"]["train_end"],
                                  "validation_end": parent["cutoffs_seconds"]["validation_end"]},
              "excluded_missing_label_pairs": {"policy_train": len(train_queries["target_tokens"]) - len(train_pairs["rewards"]),
                                                "policy_validation": len(validation_queries["target_tokens"]) - len(validation["rewards"])},
              "partitions": summaries, "label_scope": "Observed listen pairs only; mean of per-event min(played_ratio_pct,100)/100, stored as float32.",
              "test_labels_read": False, "test_evaluated": False, "reward_model_trained": False, "rl_trained": False}
    write_json(output_dir / "manifest.json", result)
    return result


def validate_rewards(arrays: dict, catalog_size: int) -> None:
    count = len(arrays["rewards"])
    if not count or any(len(values) != count for values in arrays.values()):
        raise ValueError("Empty or inconsistent reward arrays.")
    if not np.all(arrays["history_max_timestamps"] < arrays["timestamps"]):
        raise ValueError("Reward context contains a target-time event.")
    if not np.all(np.isfinite(arrays["rewards"])) or np.any((arrays["rewards"] < 0) | (arrays["rewards"] > 1)):
        raise ValueError("Reward labels must be finite and in [0, 1].")
    if np.any((arrays["candidate_tokens"] < FIRST_ITEM) | (arrays["candidate_tokens"] >= catalog_size + FIRST_ITEM)):
        raise ValueError("Reward candidate is outside the frozen catalog.")
    if np.any((arrays["lengths"] < 2) | (arrays["lengths"] > arrays["input_ids"].shape[1])):
        raise ValueError("Invalid reward context length.")
    if not np.all(arrays["input_ids"][:, 0] == 1) or np.any((arrays["input_ids"] < 0) | (arrays["input_ids"] >= catalog_size + FIRST_ITEM)):
        raise ValueError("Invalid reward context token.")


def load_reward_partition(data_dir: Path, split: str) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / f"{split}.npz"
    if sha256(path) != manifest["partitions"][split]["sha256"]:
        raise ValueError(f"Reward partition changed: {split}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    validate_rewards(arrays, manifest["catalog_size"])
    cutoff = manifest["cutoffs_seconds"]
    lower, upper = {"train": (-1, cutoff["reward_train_end"]),
                    "selection": (cutoff["reward_train_end"], cutoff["selection_end"]),
                    "validation": (cutoff["selection_end"], cutoff["validation_end"])}[split]
    if np.any((arrays["timestamps"] <= lower) | (arrays["timestamps"] > upper)):
        raise ValueError(f"Reward temporal partition violation: {split}")
    return arrays
