from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch

from recsys_rl.data import load_config, load_partition, sha256, write_json
from recsys_rl.evaluation import targets_for
from recsys_rl.model import select_device
from recsys_rl.reward import reward_source_hashes
from recsys_rl.reward_data import load_reward_partition


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_mixed_inputs(context: dict) -> None:
    if reward_source_hashes() != context["frozen"]["source_files"]:
        raise ValueError("Mixed-objective source changed.")
    for path, expected in context["frozen"]["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Mixed-objective input changed: {path}")


def load_mixed_context(config_path: Path, protocol_path: Path, output_dir: Path, holdout_dir: Path,
                       requested_device: str, resume: bool) -> tuple[dict, dict]:
    config_path, protocol_path, output_dir, holdout_dir = (path.resolve() for path in (config_path, protocol_path, output_dir, holdout_dir))
    config = read_json(config_path)
    pilot_path = (config_path.parent / config["pilot_report"]).resolve()
    final_path = (config_path.parent / config["final_report"]).resolve()
    pilot, final = read_json(pilot_path), read_json(final_path)
    for name, expected in final["provenance"]["frozen"]["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical computation changed: {name}")
    old_config_path = Path(pilot["paths"]["config_path"])
    old_config = read_json(old_config_path)
    data_dir, reward_data_dir = Path(pilot["paths"]["data_dir"]), Path(pilot["paths"]["reward_data_dir"])
    reward_dir = (old_config_path.parent / old_config["reward_output_dir"]).resolve()
    reward_report = read_json((old_config_path.parent / old_config["reward_report"]).resolve())
    signal = read_json((old_config_path.parent / old_config["signal_report"]).resolve())
    cache_root = (config_path.parent / config["pilot_output_dir"]).resolve() / "cache"
    cache_manifest = read_json(cache_root / "manifest.json")
    if cache_manifest != pilot["cache"]:
        raise ValueError("Original pilot cache manifest differs.")
    baseline = final["provenance"]["models"]["baseline"]
    inputs = {str(Path(path).resolve()): expected for path, expected in final["provenance"]["frozen"]["inputs"].items()}
    for path in (config_path, protocol_path, pilot_path, final_path, cache_root / "manifest.json", data_dir / "listen-events.parquet"):
        inputs[str(path)] = sha256(path)
    for name, value in cache_manifest["artifacts"].items():
        inputs[str(cache_root / name)] = value["sha256"]
    frozen = {"inputs": inputs, "source_files": reward_source_hashes()}
    context = {"config": config, "frozen": frozen, "output": output_dir, "holdout_dir": holdout_dir,
               "paths": {"config": str(config_path), "protocol": str(protocol_path), "holdout_dir": str(holdout_dir)},
               "data_dir": data_dir, "reward_data_dir": reward_data_dir, "reward_dir": reward_dir,
               "reward_report": reward_report, "policy_checkpoint": Path(baseline["path"]),
               "policy_config": load_config((old_config_path.parent / old_config["policy_config"]).resolve()),
               "catalog": read_json(data_dir / "catalog.json"), "manifest": read_json(data_dir / "manifest.json"),
               "source": Path(read_json(reward_data_dir / "manifest.json")["source_path"]).resolve(),
               "calibration": signal["calibration"], "device": select_device(requested_device)}
    verify_mixed_inputs(context)
    plan_path = output_dir / "plan.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError("Mixed-objective output exists; use --resume with unchanged inputs or --audit.")
        plan = read_json(plan_path)
        if plan["frozen"] != frozen or plan["paths"] != context["paths"]:
            raise ValueError("Mixed-objective plan or paths changed.")
    else:
        plan = {"config": config, "frozen": frozen, "paths": context["paths"],
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "historical_test_reevaluated": False}
        write_json(plan_path, plan)
    context["plan"] = plan
    torch.set_num_threads(4)
    caches = {}
    for split in ("train", "validation"):
        with np.load(cache_root / f"{split}.npz", allow_pickle=False) as saved:
            caches[split] = dict(saved)
    original_train = load_partition(data_dir, "train")
    indices = caches["train"]["original_indices"]
    selected_targets = [targets_for(original_train, int(index)) for index in indices]
    context["train"] = {name: original_train[name][indices] for name in ("input_ids", "lengths", "user_ids", "timestamps", "history_max_timestamps")}
    context["train"].update(target_tokens=np.concatenate(selected_targets), target_offsets=np.r_[0, np.cumsum([len(row) for row in selected_targets])])
    context["validation"] = load_partition(data_dir, "validation")
    for split in ("train", "validation"):
        for key in ("user_ids", "timestamps"):
            if not np.array_equal(context[split][key], caches[split][key]):
                raise ValueError("Mixed training or validation population differs from the frozen cache.")
    train = load_reward_partition(reward_data_dir, "train")
    context["reward_train"], context["known_users"] = train, np.unique(train["user_ids"])
    context["observed_validation"] = load_reward_partition(reward_data_dir, "validation")
    context["counts"] = np.bincount(train["candidate_tokens"] - 3, minlength=len(context["catalog"]["items"]))
    context["personal"] = {split: cache["personal_mean"] for split, cache in caches.items()}
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan for item in context["catalog"]["items"]])
    known = durations[np.isfinite(durations) & (durations > 0)]
    cap = float(np.quantile(known, old_config["reward"]["duration_cap_catalog_quantile"]))
    if cap != pilot["cache"]["duration_cap_seconds"]:
        raise ValueError("Original duration cap changed.")
    context["duration_scale"] = np.minimum(np.where(np.isfinite(durations) & (durations > 0), durations, np.median(known)), cap) / cap
    own_cache = output_dir / "cache"
    own_cache.mkdir(exist_ok=True)
    destination = own_cache / "rollout-inputs.npz"
    expected = cache_manifest["artifacts"]["rollout-inputs.npz"]["sha256"]
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError("Common rollout input changed.")
    else:
        shutil.copyfile(cache_root / "rollout-inputs.npz", destination)
    return context, caches


def prepare_fresh_holdout(context: dict) -> dict:
    root, settings = context["holdout_dir"], context["config"]["fresh_holdout"]
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["preparation_frozen"] != context["frozen"]:
            raise ValueError("Fresh holdout preparation differs.")
        for name, artifact in manifest["artifacts"].items():
            if sha256(root / name) != artifact["sha256"]:
                raise ValueError("Fresh holdout artifact changed.")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Incomplete fresh holdout must be preserved before retrying.")
    cuts, old_settings = context["manifest"]["cutoffs_seconds"], context["manifest"]["config"]
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    try:
        connection.read_parquet(str(context["source"])).create_view("raw_events")
        connection.execute("""CREATE VIEW listens AS SELECT uid, item_id, timestamp FROM raw_events
            WHERE event_type = 'listen' AND uid IS NOT NULL AND item_id IS NOT NULL AND timestamp IS NOT NULL""")
        excluded = connection.execute("SELECT DISTINCT user_id FROM read_parquet(?) ORDER BY user_id", [str(context["data_dir"] / "listen-events.parquet")]).fetchnumpy()["user_id"]
        reconstructed = connection.execute("""SELECT uid FROM listens WHERE timestamp <= ? GROUP BY uid HAVING count(*) >= ?
            ORDER BY hash(uid, ?), uid LIMIT ?""", [cuts["train_end"], old_settings["data"]["minimum_train_events"], old_settings["seed"], old_settings["data"]["max_users"]]).fetchnumpy()["uid"]
        if not np.array_equal(np.sort(reconstructed), excluded):
            raise ValueError("Original full user population does not reproduce from the raw source.")
        connection.register("excluded_users", {"user_id": excluded})
        users = connection.execute("""SELECT uid FROM listens l LEFT JOIN excluded_users x ON l.uid = x.user_id
            WHERE timestamp <= ? AND x.user_id IS NULL GROUP BY uid HAVING count(*) >= ?
            ORDER BY hash(uid, ?), uid LIMIT ?""", [cuts["train_end"], settings["minimum_train_events"], settings["sampling_seed"], settings["users"]]).fetchnumpy()["uid"]
        if len(users) != settings["users"] or np.intersect1d(users, excluded).size:
            raise ValueError("Cannot prepare the requested disjoint user cohort.")
        connection.register("new_users", {"user_id": users})
        connection.execute("""CREATE TEMP TABLE events AS SELECT uid AS user_id, timestamp, item_id
            FROM listens JOIN new_users ON uid = new_users.user_id ORDER BY uid, timestamp, item_id""")
        connection.register("catalog", {"item_id": np.asarray([item["item_id"] for item in context["catalog"]["items"]], dtype=np.uint32)})
        queries = connection.execute("""SELECT user_id, timestamp FROM
            (SELECT DISTINCT user_id, timestamp FROM events JOIN catalog USING (item_id) WHERE timestamp > ? AND timestamp <= ?)
            ORDER BY hash(user_id, timestamp, ?), user_id, timestamp LIMIT ?""", [cuts["validation_end"], cuts["data_end"], settings["sampling_seed"], settings["max_queries"]]).fetchnumpy()
        events = connection.execute("SELECT * FROM events ORDER BY user_id, timestamp, item_id").fetchnumpy()
    finally:
        connection.close()
    item_ids = np.asarray([item["item_id"] for item in context["catalog"]["items"]], dtype=np.uint32)
    order = np.argsort(item_ids)
    positions = np.searchsorted(item_ids[order], events["item_id"])
    matches = (positions < len(item_ids)) & (item_ids[order][np.minimum(positions, len(item_ids) - 1)] == events["item_id"])
    tokens = np.full(len(positions), 2, dtype=np.int32)
    tokens[matches] = order[positions[matches]] + 3
    event_users, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(user): (int(start), int(start + count)) for user, start, count in zip(event_users, starts, counts)}
    inputs, lengths, targets, offsets, uids, times, previous = [], [], [], [0], [], [], []
    skipped = 0
    history_length = context["policy_config"]["data"]["history_length"]
    for index in np.lexsort((queries["timestamp"], queries["user_id"])):
        uid, timestamp = int(queries["user_id"][index]), int(queries["timestamp"][index])
        start, end = boundaries[uid]
        user_times = events["timestamp"][start:end]
        stop = start + int(np.searchsorted(user_times, timestamp, side="left"))
        right = start + int(np.searchsorted(user_times, timestamp, side="right"))
        if stop - start < context["policy_config"]["data"]["minimum_history"]:
            skipped += 1
            continue
        current = np.unique(tokens[stop:right])
        current = current[current >= 3]
        if not current.size:
            raise ValueError("A fresh holdout query has no in-catalog target.")
        history = tokens[max(start, stop - history_length):stop]
        row = np.zeros(history_length + 1, dtype=np.int32)
        row[0], row[1:len(history) + 1] = 1, history
        inputs.append(row)
        lengths.append(len(history) + 1)
        targets.extend(current.tolist())
        offsets.append(len(targets))
        uids.append(uid)
        times.append(timestamp)
        previous.append(int(events["timestamp"][stop - 1]))
    if not inputs:
        raise ValueError("Fresh holdout has no eligible queries.")
    arrays = {"input_ids": np.asarray(inputs, dtype=np.int32), "lengths": np.asarray(lengths, dtype=np.int16),
              "target_tokens": np.asarray(targets, dtype=np.int32), "target_offsets": np.asarray(offsets, dtype=np.int64),
              "user_ids": np.asarray(uids, dtype=np.uint32), "timestamps": np.asarray(times, dtype=np.uint32),
              "history_max_timestamps": np.asarray(previous, dtype=np.uint32)}
    if not np.all(arrays["history_max_timestamps"] < arrays["timestamps"]):
        raise ValueError("Fresh holdout contains a target-time event in history.")
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "queries.npz", **arrays)
    np.savez_compressed(root / "users.npz", users=np.sort(users), excluded_users=excluded)
    manifest = {"experiment": "mixed-holdout-v1", "preparation_frozen": context["frozen"], "source_commit": context["plan"]["source_commit"],
                "selected_users": len(users), "excluded_original_users": len(excluded), "user_overlap": 0,
                "queries": len(inputs), "query_users": len(np.unique(arrays["user_ids"])), "targets": len(targets), "skipped_short_histories": skipped,
                "timestamp_min": int(arrays["timestamps"].min()), "timestamp_max": int(arrays["timestamps"].max()),
                "response_labels_read": False, "evaluated": False, "catalog_size": len(item_ids),
                "artifacts": {name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size} for name in ("queries.npz", "users.npz")}}
    verify_mixed_inputs(context)
    write_json(manifest_path, manifest)
    return manifest
