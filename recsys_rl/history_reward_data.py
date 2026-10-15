from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch

from recsys_rl.data import sha256, write_json
from recsys_rl.model import select_device
from recsys_rl.recency_data import read_json
from recsys_rl.reward import reward_source_hashes
from recsys_rl.reward_data import attach_rewards, load_reward_partition, validate_rewards
from recsys_rl.reward_model import duration_features


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def verify(context: dict) -> None:
    frozen = context["plan"]["frozen"]
    if reward_source_hashes() != frozen["source_files"]:
        raise ValueError("History-reward source changed.")
    for name, expected in frozen["inputs"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def load_context(config_path: Path, protocol: Path, output: Path, device: str, resume: bool) -> dict:
    config_path, protocol, output = [p.resolve() for p in (config_path, protocol, output)]
    config = read_json(config_path)
    resolve = lambda key: (config_path.parent / config[key]).resolve()
    reference = read_json(resolve("reference_report"))
    execution = read_json(resolve("reference_execution"))
    if sha256(resolve("reference_report")) != execution["result_sha256"]:
        raise ValueError("Recency result changed.")
    frozen = reference["provenance"]["frozen"]
    for name, expected in frozen["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical computation changed: {name}")
    inputs = dict(frozen["inputs"])
    for name, expected in inputs.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Historical artifact changed: {name}")
    reward_dir, previous = resolve("reward_data_dir"), resolve("previous_recency_dir")
    reward_manifest = read_json(reward_dir / "manifest.json")
    data_dir = Path(reward_manifest["policy_data_dir"])
    source = Path(reward_manifest["source_path"])
    for p in (config_path, protocol, resolve("reference_report"), resolve("reference_execution"),
              reward_dir / "manifest.json", previous / "manifest.json", previous / "users.npz",
              config_path.parent / "reward.json", config_path.parent.parent / "experiments" / "reward-signal-result.json"):
        inputs[str(p)] = sha256(p)
    recency_root = Path(reference["runs"]["global-16/42"]["checkpoint"]).parents[3]
    for p, expected in ((recency_root / "cache/validation.npy", reference["cache"]["artifacts"]["validation.npy"]),
                        (recency_root / "holdout/baseline.npy", reference["holdout"]["artifacts"]["baseline.npy"]),
                        (previous / "queries.npz", reference["preparation"]["artifacts"]["queries.npz"]),
                        (Path(reference["runs"]["global-16/42"]["checkpoint"]), reference["runs"]["global-16/42"]["checkpoint_sha256"])):
        if sha256(p) != expected:
            raise ValueError("Historical recency artifact changed.")
        inputs[str(p)] = expected
    own_frozen = {"inputs": inputs, "source_files": reward_source_hashes()}
    paths = {"config": str(config_path), "protocol": str(protocol), "output": str(output), "data": str(resolve("new_data_dir"))}
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError("History-reward output exists; use --resume or --audit.")
        plan = read_json(output / "plan.json")
        if plan["frozen"] != own_frozen or plan["paths"] != paths:
            raise ValueError("History-reward frozen plan differs.")
    else:
        plan = {"config": config, "frozen": own_frozen, "paths": paths,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "plan.json", plan)
    context = {"config": config, "plan": plan, "output": output, "new_data_dir": resolve("new_data_dir"),
               "reference": reference, "reference_execution": execution, "previous": previous, "source": source,
               "data_dir": data_dir, "reward_data_dir": reward_dir, "reward_manifest": reward_manifest,
               "manifest": read_json(data_dir / "manifest.json"), "catalog": read_json(data_dir / "catalog.json"),
               "device": select_device(device), "recency_root": recency_root,
               "old_reward_config": read_json(config_path.parent / "reward.json"),
               "calibration": read_json(config_path.parent.parent / "experiments" / "reward-signal-result.json")["calibration"]}
    torch.set_num_threads(config["execution"]["threads"])
    context["train"] = load_reward_partition(reward_dir, "train")
    context["selection"] = load_reward_partition(reward_dir, "selection")
    verify(context)
    return context


def connection(context: dict):
    db = duckdb.connect()
    db.execute("SET threads = 4")
    db.execute("SET memory_limit = '2GB'")
    db.read_parquet(str(context["source"])).create_view("raw_events")
    db.execute("""CREATE VIEW listens AS SELECT uid, item_id, timestamp FROM raw_events
        WHERE event_type = 'listen' AND uid IS NOT NULL AND item_id IS NOT NULL AND timestamp IS NOT NULL""")
    return db


def build_cohort(context: dict, db, settings: dict, excluded: np.ndarray) -> tuple[dict, dict]:
    cuts = context["manifest"]["cutoffs_seconds"]
    db.register("excluded_users", {"user_id": excluded})
    users = db.execute("""SELECT uid FROM listens l LEFT JOIN excluded_users x ON l.uid=x.user_id
        WHERE timestamp<=? AND x.user_id IS NULL GROUP BY uid HAVING count(*)>=50
        ORDER BY hash(uid, ?), uid LIMIT ?""", [cuts["train_end"], settings["seed"], settings["users"]]).fetchnumpy()["uid"]
    db.unregister("excluded_users")
    if len(users) != settings["users"] or np.intersect1d(users, excluded).size:
        raise ValueError("Insufficient unused users.")
    db.register("cohort_users", {"user_id": users})
    events = db.execute("""SELECT uid AS user_id,timestamp,item_id FROM listens JOIN cohort_users ON uid=user_id
        ORDER BY uid,timestamp,item_id""").fetchnumpy()
    items = np.asarray([item["item_id"] for item in context["catalog"]["items"]], dtype=np.uint32)
    db.register("catalog_items", {"item_id": items})
    lower, upper = (cuts["train_end"], cuts["validation_end"]) if settings["period"] == "validation" else (cuts["validation_end"], cuts["data_end"])
    queries = db.execute("""SELECT user_id,timestamp FROM
        (SELECT DISTINCT uid AS user_id,timestamp FROM listens JOIN cohort_users ON uid=user_id JOIN catalog_items USING(item_id)
         WHERE timestamp>? AND timestamp<=?) ORDER BY hash(user_id,timestamp,?),user_id,timestamp LIMIT ?""",
        [lower, upper, settings["seed"], settings["queries"]]).fetchnumpy()
    db.unregister("cohort_users"); db.unregister("catalog_items")
    order = np.argsort(items)
    positions = np.searchsorted(items[order], events["item_id"])
    valid = (positions < len(items)) & (items[order][np.minimum(positions, len(items)-1)] == events["item_id"])
    tokens = np.full(len(positions), 2, dtype=np.int32)
    tokens[valid] = order[positions[valid]]+3
    unique, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(u): (int(s), int(s+c)) for u, s, c in zip(unique, starts, counts)}
    inputs, lengths, targets, offsets, uids, times, maximum = [], [], [], [0], [], [], []
    for i in np.lexsort((queries["timestamp"], queries["user_id"])):
        uid, timestamp = int(queries["user_id"][i]), int(queries["timestamp"][i])
        start, end = boundaries[uid]
        stop = start+np.searchsorted(events["timestamp"][start:end], timestamp, side="left")
        right = start+np.searchsorted(events["timestamp"][start:end], timestamp, side="right")
        if stop-start < 5:
            continue
        current = np.unique(tokens[stop:right]); current = current[current >= 3]
        if not current.size:
            raise ValueError("New query has no catalog target.")
        history = tokens[max(start, stop-64):stop]
        row = np.zeros(65, dtype=np.int32); row[0] = 1; row[1:len(history)+1] = history
        inputs.append(row); lengths.append(len(history)+1); targets.extend(current.tolist()); offsets.append(len(targets))
        uids.append(uid); times.append(timestamp); maximum.append(int(events["timestamp"][stop-1]))
    if not inputs:
        raise ValueError("Empty new cohort.")
    arrays = {"input_ids": np.asarray(inputs, dtype=np.int32), "lengths": np.asarray(lengths, dtype=np.int16),
              "target_tokens": np.asarray(targets, dtype=np.int32), "target_offsets": np.asarray(offsets, dtype=np.int64),
              "user_ids": np.asarray(uids, dtype=np.uint32), "timestamps": np.asarray(times, dtype=np.uint32),
              "history_max_timestamps": np.asarray(maximum, dtype=np.uint32)}
    if not np.all(arrays["history_max_timestamps"] < arrays["timestamps"]):
        raise ValueError("New cohort has a target-time leak.")
    return arrays, {"users": np.sort(users), "excluded_users": excluded}


def prepare(context: dict) -> dict:
    root = context["new_data_dir"]
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if manifest["frozen"] != context["plan"]["frozen"]:
            raise ValueError("History preparation plan changed.")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Prepared history artifact changed.")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete history preparation.")
    db = connection(context)
    try:
        original = db.execute("SELECT DISTINCT user_id FROM read_parquet(?) ORDER BY user_id", [str(context["data_dir"] / "listen-events.parquet")]).fetchnumpy()["user_id"]
        recency_users = load_npz(context["previous"] / "users.npz")
        prior_manifest = read_json(context["previous"] / "manifest.json")
        if sha256(context["previous"] / "users.npz") != prior_manifest["artifacts"]["users.npz"]:
            raise ValueError("Third complete cohort changed.")
        excluded = np.union1d(recency_users["users"], recency_users["excluded_users"])
        if len(excluded) != 3000 or not np.all(np.isin(original, excluded)):
            raise ValueError("Expected three complete historical populations.")
        development, dev_users = build_cohort(context, db, context["config"]["reward_development"], excluded)
        final, final_users = build_cohort(context, db, context["config"]["final_reserve"], np.union1d(excluded, dev_users["users"]))
    finally:
        db.close()
    root.mkdir(parents=True, exist_ok=True)
    for name, arrays in (("development", development), ("development-users", dev_users), ("final", final), ("final-users", final_users)):
        np.savez_compressed(root / f"{name}.npz", **arrays)
    manifest = {"frozen": context["plan"]["frozen"], "source_commit": context["plan"]["source_commit"],
                "development": {"selected_users": len(dev_users["users"]), "excluded_users": len(excluded), "queries": len(development["lengths"]), "query_users": len(np.unique(development["user_ids"]))},
                "final_reserve": {"selected_users": len(final_users["users"]), "excluded_users": len(final_users["excluded_users"]), "queries": len(final["lengths"]), "query_users": len(np.unique(final["user_ids"]))},
                "overlap": 0, "response_labels_read": False, "final_evaluated": False,
                "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}}
    write_json(root / "manifest.json", manifest)
    verify(context)
    return manifest


def attach_observed(context: dict, arrays: dict) -> dict:
    db = connection(context)
    try:
        db.execute("CREATE VIEW reward_source AS SELECT * FROM raw_events")
        result = attach_rewards(db, arrays, context["catalog"])
    finally:
        db.close()
    validate_rewards(result, len(context["catalog"]["items"]))
    return result


def history_features(context: dict, arrays: dict, training: bool) -> dict:
    users = np.unique(arrays["user_ids"])
    db = connection(context)
    try:
        db.register("requested_users", {"user_id": users})
        events = db.execute("""SELECT uid AS user_id,timestamp,item_id,
            avg(CASE WHEN played_ratio_pct IS NOT NULL THEN least(played_ratio_pct,100)/100.0 END) AS response
            FROM raw_events JOIN requested_users ON uid=user_id WHERE event_type='listen' AND timestamp<?
            GROUP BY uid,timestamp,item_id ORDER BY uid,timestamp,item_id""", [int(arrays["timestamps"].max())]).fetchnumpy()
    finally:
        db.close()
    responses = np.ma.filled(events["response"], np.nan).astype(np.float64)
    items = np.asarray([item["item_id"] for item in context["catalog"]["items"]], dtype=np.uint32)
    ordering = np.argsort(items); positions = np.searchsorted(items[ordering], events["item_id"])
    matches = (positions < len(items)) & (items[ordering][np.minimum(positions, len(items)-1)] == events["item_id"])
    tokens = np.full(len(positions), 2, dtype=np.int32); tokens[matches] = ordering[positions[matches]]+3
    unique, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(u): (int(s), int(s+c)) for u, s, c in zip(unique, starts, counts)}
    train = context["train"]; labels = train["rewards"].astype(np.float64)
    totals = np.bincount(train["candidate_tokens"], weights=labels, minlength=len(items)+3)
    frequencies = np.bincount(train["candidate_tokens"], minlength=len(items)+3)
    label_total = labels.sum()
    global_mean = labels.mean(); settings = context["config"]["history"]
    durations = duration_features(context["catalog"])
    features = np.empty((len(arrays["rewards"]), 16), dtype=np.float32)
    maximum_times = np.empty(len(features), dtype=np.uint32)
    for i, (uid, timestamp, candidate) in enumerate(zip(arrays["user_ids"], arrays["timestamps"], arrays["candidate_tokens"])):
        start, end = boundaries[int(uid)]
        stop = start+int(np.searchsorted(events["timestamp"][start:end], timestamp, side="left"))
        lower = max(start, stop-settings["events"])
        values, history = responses[lower:stop], tokens[lower:stop]
        valid = np.isfinite(values); observed = values[valid]
        prior = (label_total-float(arrays["rewards"][i]))/(len(labels)-1) if training else global_mean
        user = (observed.sum()+settings["user_smoothing"]*prior)/(len(observed)+settings["user_smoothing"])
        item_sum = totals[candidate]-float(arrays["rewards"][i]) if training else totals[candidate]
        item_count = frequencies[candidate]-1 if training else frequencies[candidate]
        item = (item_sum+settings["item_smoothing"]*prior)/(item_count+settings["item_smoothing"])
        occurrences = history == candidate; candidate_observed = values[occurrences & valid]
        candidate_prior = (candidate_observed.sum()+settings["candidate_smoothing"]*user)/(len(candidate_observed)+settings["candidate_smoothing"])
        def short_mean(size: int) -> float:
            selected = values[-size:]; selected = selected[np.isfinite(selected)]
            return float(selected.mean()) if selected.size else float(user)
        feature = [user, short_mean(4), short_mean(16), float(observed.std()) if observed.size else 0.0,
                   float((observed >= .9).mean()) if observed.size else user,
                   float((observed == 0).mean()) if observed.size else 0.0, len(observed)/64,
                   float(np.mean(arrays["input_ids"][i, 1:int(arrays["lengths"][i])] == 2)), item, candidate_prior,
                   float(occurrences.sum()/64), float((len(history)-1-np.flatnonzero(occurrences)[-1])/64) if occurrences.any() else 1.0,
                   float(candidate_observed[-1]) if candidate_observed.size else user,
                   float(durations[candidate, 0]/3), float(durations[candidate, 1]), float(occurrences.any())]
        features[i] = feature
        maximum_times[i] = int(events["timestamp"][stop-1]) if stop > start else 0
    if not np.isfinite(features).all() or not np.all(maximum_times < arrays["timestamps"]):
        raise ValueError("Invalid response features or temporal leakage.")
    return {"features": features, "response_history_max_timestamps": maximum_times}
