from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch

from recsys_rl.data import load_partition, sha256, write_json
from recsys_rl.model import select_device
from recsys_rl.reward import reward_source_hashes


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(context: dict) -> None:
    if reward_source_hashes() != context["plan"]["frozen"]["source_files"]:
        raise ValueError("Recency source changed.")
    for path, expected in context["plan"]["frozen"]["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Recency frozen input changed: {path}")


def load_context(config_path: Path, protocol_path: Path, output: Path, holdout: Path,
                 device: str, resume: bool) -> dict:
    config_path, protocol_path, output, holdout = [p.resolve() for p in (config_path, protocol_path, output, holdout)]
    config = read_json(config_path)
    resolve = lambda key: (config_path.parent / config[key]).resolve()
    historical = read_json(resolve("historical_report"))
    old_frozen = historical["provenance"]["frozen"]
    for name, expected in old_frozen["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical code changed: {name}")
    inputs = dict(old_frozen["inputs"])
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Historical artifact changed: {path}")
    data_dir, previous = resolve("data_dir"), resolve("previous_holdout")
    for p in (config_path, protocol_path, resolve("historical_report"), resolve("baseline_config"),
              resolve("baseline_checkpoint"), data_dir / "train.npz", data_dir / "validation.npz", data_dir / "manifest.json",
              data_dir / "catalog.json", data_dir / "listen-events.parquet", previous / "users.npz", previous / "manifest.json"):
        inputs[str(p)] = sha256(p)
    frozen = {"inputs": inputs, "source_files": reward_source_hashes()}
    paths = {"config": str(config_path), "protocol": str(protocol_path), "holdout": str(holdout)}
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError("Recency output exists; use --resume or --audit with unchanged inputs.")
        plan = read_json(output / "plan.json")
        if plan["frozen"] != frozen or plan["paths"] != paths:
            raise ValueError("Recency plan differs from current inputs.")
    else:
        plan = {"config": config, "frozen": frozen, "paths": paths,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "plan.json", plan)
    context = {"config": config, "plan": plan, "output": output, "holdout_dir": holdout,
               "data_dir": data_dir, "previous": previous, "manifest": read_json(data_dir / "manifest.json"),
               "catalog": read_json(data_dir / "catalog.json"), "policy_config": read_json(resolve("baseline_config")),
               "policy_checkpoint": resolve("baseline_checkpoint"), "device": select_device(device)}
    # Use the raw path already frozen by the historical experiment.
    sources = [Path(p) for p in inputs if Path(p).name == "yambda-50m-multi_event.parquet"]
    if len(sources) != 1:
        raise ValueError("Expected one frozen raw Yambda source.")
    context["source"] = sources[0]
    torch.set_num_threads(config["execution"]["threads"])
    context["train"] = load_partition(data_dir, "train")
    context["validation"] = load_partition(data_dir, "validation")
    verify_inputs(context)
    return context


def build_holdout(context: dict) -> tuple[dict, dict, dict]:
    settings, cuts = context["config"]["holdout"], context["manifest"]["cutoffs_seconds"]
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    try:
        connection.read_parquet(str(context["source"])).create_view("raw_events")
        connection.execute("""CREATE VIEW listens AS SELECT uid, item_id, timestamp FROM raw_events
            WHERE event_type = 'listen' AND uid IS NOT NULL AND item_id IS NOT NULL AND timestamp IS NOT NULL""")
        original = connection.execute("SELECT DISTINCT user_id FROM read_parquet(?) ORDER BY user_id", [str(context["data_dir"] / "listen-events.parquet")]).fetchnumpy()["user_id"]
        with np.load(context["previous"] / "users.npz", allow_pickle=False) as saved:
            previous = saved["users"].copy()
            if not np.array_equal(saved["excluded_users"], original):
                raise ValueError("Previous cohort does not exclude the complete original cohort.")
        previous_manifest = read_json(context["previous"] / "manifest.json")
        if sha256(context["previous"] / "users.npz") != previous_manifest["artifacts"]["users.npz"]["sha256"]:
            raise ValueError("Previous complete cohort differs from its historical manifest.")
        excluded = np.union1d(original, previous)
        if len(excluded) != len(original) + len(previous):
            raise ValueError("Historical cohorts overlap.")
        connection.register("excluded_users", {"user_id": excluded})
        users = connection.execute("""SELECT uid FROM listens l LEFT JOIN excluded_users x ON l.uid = x.user_id
            WHERE timestamp <= ? AND x.user_id IS NULL GROUP BY uid HAVING count(*) >= ?
            ORDER BY hash(uid, ?), uid LIMIT ?""", [cuts["train_end"], settings["minimum_train_events"], settings["sampling_seed"], settings["users"]]).fetchnumpy()["uid"]
        if len(users) != settings["users"] or np.intersect1d(users, excluded).size:
            raise ValueError("Insufficient disjoint eligible users.")
        connection.register("new_users", {"user_id": users})
        connection.execute("""CREATE TEMP TABLE events AS SELECT uid AS user_id, timestamp, item_id
            FROM listens JOIN new_users ON uid = new_users.user_id ORDER BY uid, timestamp, item_id""")
        item_ids = np.asarray([item["item_id"] for item in context["catalog"]["items"]], dtype=np.uint32)
        connection.register("catalog", {"item_id": item_ids})
        queries = connection.execute("""SELECT user_id, timestamp FROM
            (SELECT DISTINCT user_id, timestamp FROM events JOIN catalog USING (item_id) WHERE timestamp > ? AND timestamp <= ?)
            ORDER BY hash(user_id, timestamp, ?), user_id, timestamp LIMIT ?""", [cuts["validation_end"], cuts["data_end"], settings["sampling_seed"], settings["max_queries"]]).fetchnumpy()
        events = connection.execute("SELECT * FROM events ORDER BY user_id, timestamp, item_id").fetchnumpy()
    finally:
        connection.close()
    order = np.argsort(item_ids)
    positions = np.searchsorted(item_ids[order], events["item_id"])
    matches = (positions < len(item_ids)) & (item_ids[order][np.minimum(positions, len(item_ids) - 1)] == events["item_id"])
    tokens = np.full(len(positions), 2, dtype=np.int32)
    tokens[matches] = order[positions[matches]] + 3
    event_users, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(u): (int(s), int(s+c)) for u, s, c in zip(event_users, starts, counts)}
    inputs, lengths, targets, offsets, uids, times, previous_times = [], [], [], [0], [], [], []
    history_length = context["policy_config"]["data"]["history_length"]
    skipped = 0
    for index in np.lexsort((queries["timestamp"], queries["user_id"])):
        uid, timestamp = int(queries["user_id"][index]), int(queries["timestamp"][index])
        start, end = boundaries[uid]
        user_times = events["timestamp"][start:end]
        stop = start + int(np.searchsorted(user_times, timestamp, side="left"))
        right = start + int(np.searchsorted(user_times, timestamp, side="right"))
        if stop-start < context["policy_config"]["data"]["minimum_history"]:
            skipped += 1
            continue
        current = np.unique(tokens[stop:right])
        current = current[current >= 3]
        if not current.size:
            raise ValueError("Holdout query has no target.")
        history = tokens[max(start, stop-history_length):stop]
        row = np.zeros(history_length+1, dtype=np.int32)
        row[0], row[1:len(history)+1] = 1, history
        inputs.append(row); lengths.append(len(history)+1)
        targets.extend(current.tolist()); offsets.append(len(targets))
        uids.append(uid); times.append(timestamp); previous_times.append(int(events["timestamp"][stop-1]))
    if not inputs:
        raise ValueError("Holdout has no queries.")
    arrays = {"input_ids": np.asarray(inputs, dtype=np.int32), "lengths": np.asarray(lengths, dtype=np.int16),
              "target_tokens": np.asarray(targets, dtype=np.int32), "target_offsets": np.asarray(offsets, dtype=np.int64),
              "user_ids": np.asarray(uids, dtype=np.uint32), "timestamps": np.asarray(times, dtype=np.uint32),
              "history_max_timestamps": np.asarray(previous_times, dtype=np.uint32)}
    if not np.all(arrays["history_max_timestamps"] < arrays["timestamps"]):
        raise ValueError("Target-time history leak.")
    cohort = {"users": np.sort(users), "excluded_users": excluded}
    description = {"selected_users": len(users), "excluded_users": len(excluded), "user_overlap": 0,
                   "queries": len(inputs), "query_users": len(np.unique(arrays["user_ids"])), "targets": len(targets),
                   "skipped_short_histories": skipped, "response_labels_read": False, "evaluated": False}
    return arrays, cohort, description


def prepare_holdout(context: dict) -> dict:
    root = context["holdout_dir"]
    if (root / "manifest.json").exists():
        result = read_json(root / "manifest.json")
        if result["frozen"] != context["plan"]["frozen"]:
            raise ValueError("Holdout preparation changed.")
        for name, expected in result["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Holdout artifact changed.")
        return result
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete holdout before retrying.")
    arrays, cohort, description = build_holdout(context)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "queries.npz", **arrays)
    np.savez_compressed(root / "users.npz", **cohort)
    result = {**description, "frozen": context["plan"]["frozen"], "source_commit": context["plan"]["source_commit"],
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "artifacts": {name: sha256(root / name) for name in ("queries.npz", "users.npz")}}
    write_json(root / "manifest.json", result)
    verify_inputs(context)
    return result
