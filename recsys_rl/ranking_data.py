from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recsys_rl.data import sha256, write_json
from recsys_rl.history_reward_data import attach_observed, connection, history_features, load_npz
from recsys_rl.recency_data import read_json
from recsys_rl.reward import reward_source_hashes
from recsys_rl.reward_data import load_reward_partition


def verify(context: dict) -> None:
    frozen = context["plan"]["frozen"]
    if reward_source_hashes() != frozen["source_files"]:
        raise ValueError("Ranking source changed.")
    for path, expected in frozen["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen ranking input changed: {path}")


def load_context(config_path: Path, protocol: Path, output: Path, resume: bool) -> dict:
    config_path, protocol, output = [p.resolve() for p in (config_path, protocol, output)]
    config = read_json(config_path)
    resolve = lambda key: (config_path.parent / config[key]).resolve()
    reference = read_json(resolve("reference_report"))
    execution = read_json(resolve("reference_execution"))
    if sha256(resolve("reference_report")) != execution["result_sha256"]:
        raise ValueError("Historical reward result changed.")
    frozen = reference["provenance"]["frozen"]
    for name, expected in frozen["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical source changed: {name}")
    inputs = dict(frozen["inputs"])
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Historical input changed: {path}")
    history_dir = resolve("history_data_dir")
    for path in (config_path, protocol, resolve("reference_report"), resolve("reference_execution"),
                 resolve("dependencies"), config_path.parent.parent / "requirements-ranking.txt",
                 config_path.parent.parent / "experiments/ranking-dependencies.json",
                 history_dir / "manifest.json", history_dir / "development-users.npz", history_dir / "final-users.npz"):
        inputs[str(path)] = sha256(path)
    own_frozen = {"inputs": inputs, "source_files": reward_source_hashes()}
    paths = {"config": str(config_path), "protocol": str(protocol), "output": str(output), "data": str(resolve("new_data_dir"))}
    if output.exists() and any(output.iterdir()):
        if not resume: raise FileExistsError("Ranking output exists; use --resume or --audit.")
        plan = read_json(output / "plan.json")
        if plan["frozen"] != own_frozen or plan["paths"] != paths:
            raise ValueError("Ranking plan differs.")
    else:
        plan = {"config": config, "frozen": own_frozen, "paths": paths,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "plan.json", plan)
    data_dir = next(Path(p).parent for p in inputs if p.endswith("data/processed/mvp/manifest.json"))
    reward_dir = next(Path(p).parent for p in inputs if p.endswith("data/processed/reward-v1/manifest.json"))
    context = {"config": config, "plan": plan, "output": output, "new_data_dir": resolve("new_data_dir"),
               "history_dir": history_dir, "source": next(Path(p) for p in inputs if p.endswith("yambda-50m-multi_event.parquet")),
               "data_dir": data_dir, "catalog": read_json(data_dir / "catalog.json"),
               "manifest": read_json(data_dir / "manifest.json"), "reward_manifest": read_json(reward_dir / "manifest.json"),
               "legacy_train": load_reward_partition(reward_dir, "train"),
               "calibration_rows": load_reward_partition(reward_dir, "selection"), "device": torch.device("cpu")}
    torch.set_num_threads(config["execution"]["threads"])
    verify(context)
    return context


def choose_users(db, excluded: np.ndarray, count: int, seed: int, cutoff: int) -> np.ndarray:
    db.register("excluded_users", {"user_id": excluded})
    users = db.execute("""SELECT uid FROM listens l LEFT JOIN excluded_users x ON uid=user_id
        WHERE timestamp<=? AND x.user_id IS NULL GROUP BY uid HAVING count(*)>=50
        ORDER BY hash(uid,?),uid LIMIT ?""", [cutoff, seed, count]).fetchnumpy()["uid"]
    db.unregister("excluded_users")
    if len(users) != count or np.intersect1d(users, excluded).size:
        raise ValueError("Insufficient eligible ranking users.")
    return np.sort(users)


def queries_for(context: dict, db, users: np.ndarray, lower: int, upper: int, seed: int,
                limit: int | None = None, multi_limit: int | None = None, single_limit: int | None = None) -> dict:
    db.register("requested_users", {"user_id": users})
    items = np.asarray([i["item_id"] for i in context["catalog"]["items"]], dtype=np.uint32)
    db.register("catalog_items", {"item_id": items})
    events = db.execute("""SELECT uid AS user_id,timestamp,item_id FROM listens JOIN requested_users ON uid=user_id
        WHERE timestamp<=? ORDER BY uid,timestamp,item_id""", [upper]).fetchnumpy()
    db.execute("""CREATE TEMP TABLE eligible_queries AS SELECT uid AS user_id,timestamp,count(DISTINCT item_id) AS targets
        FROM listens JOIN requested_users ON uid=user_id JOIN catalog_items USING(item_id)
        WHERE timestamp>? AND timestamp<=? GROUP BY uid,timestamp""", [lower, upper])
    if limit is not None:
        queries = db.execute("SELECT user_id,timestamp FROM eligible_queries ORDER BY hash(user_id,timestamp,?),user_id,timestamp LIMIT ?", [seed, limit]).fetchnumpy()
    else:
        blocks = [db.execute(f"SELECT user_id,timestamp FROM eligible_queries WHERE targets {condition} ORDER BY hash(user_id,timestamp,?),user_id,timestamp LIMIT ?",
                             [seed, cap]).fetchnumpy() for condition, cap in ((">1", multi_limit), ("=1", single_limit))]
        queries = {key: np.concatenate([b[key] for b in blocks]) for key in ("user_id", "timestamp")}
    db.execute("DROP TABLE eligible_queries"); db.unregister("requested_users"); db.unregister("catalog_items")
    item_order = np.argsort(items); positions = np.searchsorted(items[item_order], events["item_id"])
    matched = (positions < len(items)) & (items[item_order][np.minimum(positions, len(items)-1)] == events["item_id"])
    tokens = np.full(len(positions), 2, dtype=np.int32); tokens[matched] = item_order[positions[matched]]+3
    unique, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(u): (int(s), int(s+c)) for u, s, c in zip(unique, starts, counts)}
    inputs, lengths, targets, offsets, uids, times, maximum = [], [], [], [0], [], [], []
    for i in np.lexsort((queries["timestamp"], queries["user_id"])):
        uid, timestamp = int(queries["user_id"][i]), int(queries["timestamp"][i])
        start, end = boundaries[uid]; history_times = events["timestamp"][start:end]
        stop = start+int(np.searchsorted(history_times, timestamp, side="left"))
        right = start+int(np.searchsorted(history_times, timestamp, side="right"))
        if stop-start < 5: continue
        current = np.unique(tokens[stop:right]); current = current[current >= 3]
        history = tokens[max(start, stop-64):stop]
        row = np.zeros(65, dtype=np.int32); row[0] = 1; row[1:len(history)+1] = history
        inputs.append(row); lengths.append(len(history)+1); targets.extend(current.tolist()); offsets.append(len(targets))
        uids.append(uid); times.append(timestamp); maximum.append(int(events["timestamp"][stop-1]))
    if not inputs: raise ValueError("Ranking cohort has no valid queries.")
    result = {"input_ids": np.asarray(inputs, dtype=np.int32), "lengths": np.asarray(lengths, dtype=np.int16),
              "target_tokens": np.asarray(targets, dtype=np.int32), "target_offsets": np.asarray(offsets, dtype=np.int64),
              "user_ids": np.asarray(uids, dtype=np.uint32), "timestamps": np.asarray(times, dtype=np.uint32),
              "history_max_timestamps": np.asarray(maximum, dtype=np.uint32)}
    if not np.all(result["history_max_timestamps"] < result["timestamps"]): raise ValueError("Ranking history leakage.")
    return result


def build(context: dict) -> dict:
    settings = context["config"]["data"]; cuts = context["manifest"]["cutoffs_seconds"]
    previous = load_npz(context["history_dir"] / "development-users.npz")
    reserve = load_npz(context["history_dir"] / "final-users.npz")
    historical_manifest = read_json(context["history_dir"] / "manifest.json")
    for name in ("development-users.npz", "final-users.npz"):
        if sha256(context["history_dir"] / name) != historical_manifest["artifacts"][name]: raise ValueError("Historical reserve changed.")
    db = connection(context)
    try:
        original = db.execute("SELECT DISTINCT user_id FROM read_parquet(?) ORDER BY user_id", [str(context["data_dir"] / "listen-events.parquet")]).fetchnumpy()["user_id"]
        excluded = np.union1d(np.union1d(previous["users"], reserve["users"]), original)
        extra = choose_users(db, excluded, settings["training_users"]-len(original), settings["additional_user_seed"], cuts["train_end"])
        training_users = np.union1d(original, extra)
        all_used = np.union1d(np.union1d(previous["excluded_users"], previous["users"]), np.union1d(training_users, reserve["users"]))
        confirmation_users = choose_users(db, all_used, settings["confirmation_users"], settings["confirmation_seed"], cuts["train_end"])
        expanded_queries = queries_for(context, db, training_users, settings["prior_end"], context["reward_manifest"]["cutoffs_seconds"]["reward_train_end"], settings["query_seed"],
                                       multi_limit=settings["multi_query_limit"], single_limit=settings["single_query_limit"])
        development_queries = queries_for(context, db, previous["users"], cuts["train_end"], cuts["validation_end"], settings["query_seed"], limit=settings["development_query_limit"])
        confirmation_queries = queries_for(context, db, confirmation_users, cuts["train_end"], cuts["validation_end"], settings["confirmation_seed"], limit=settings["confirmation_query_limit"])
        db.register("prior_users", {"user_id": original})
        db.register("catalog_items", {"item_id": np.asarray([i["item_id"] for i in context["catalog"]["items"]], dtype=np.uint32),
                                       "token": np.arange(3, len(context["catalog"]["items"])+3, dtype=np.int32)})
        prior = db.execute("""SELECT uid AS user_ids,timestamp AS timestamps,token AS candidate_tokens,
            avg(least(played_ratio_pct,100)/100.0) AS rewards FROM raw_events JOIN prior_users ON uid=user_id
            JOIN catalog_items USING(item_id) WHERE event_type='listen' AND timestamp<=? AND played_ratio_pct IS NOT NULL
            GROUP BY uid,timestamp,token ORDER BY uid,timestamp,token""", [settings["prior_end"]]).fetchnumpy()
        prior["rewards"] = prior["rewards"].astype(np.float32)
    finally: db.close()
    small = {k: v[context["legacy_train"]["timestamps"] > settings["prior_end"]] for k, v in context["legacy_train"].items()}
    if not len(small["rewards"]) or int(prior["timestamps"].max()) >= int(small["timestamps"].min()): raise ValueError("Invalid prior/train separation.")
    expanded = attach_observed(context, expanded_queries)
    development = attach_observed(context, development_queries)
    populations = {"training_users": training_users, "original_users": original, "additional_users": extra,
                   "development_users": previous["users"], "confirmation_users": confirmation_users,
                   "confirmation_excluded_users": all_used, "final_reserve_users": reserve["users"]}
    if any(np.intersect1d(training_users, populations[name]).size for name in ("development_users", "confirmation_users", "final_reserve_users")):
        raise ValueError("Training and evaluation user overlap.")
    feature_context = {**context, "train": prior}
    arrays = {"small": small, "expanded": expanded, "development": development, "calibration": context["calibration_rows"],
              "confirmation-queries": confirmation_queries, "priors": prior, "populations": populations}
    for name in ("small", "expanded", "development", "calibration"):
        arrays[f"{name}-features"] = history_features(feature_context, arrays[name], False)
    return arrays


def prepare(context: dict) -> dict:
    root = context["new_data_dir"]
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if manifest["frozen"] != context["plan"]["frozen"]: raise ValueError("Ranking preparation changed.")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected: raise ValueError("Ranking artifact changed.")
        return manifest
    if root.exists() and any(root.iterdir()): raise FileExistsError("Preserve incomplete ranking preparation.")
    values = build(context)
    root.mkdir(parents=True, exist_ok=True)
    for name, arrays in values.items(): np.savez_compressed(root / f"{name}.npz", **arrays)
    manifest = {"frozen": context["plan"]["frozen"], "source_commit": context["plan"]["source_commit"],
                "partitions": {name: {"pairs": len(values[name]["rewards"]), "queries": len(np.unique(values[name]["query_indices"])),
                                      "users": len(np.unique(values[name]["user_ids"]))} for name in ("small", "expanded", "development", "calibration")},
                "populations": {name: len(v) for name, v in values["populations"].items()},
                "prior_pairs": len(values["priors"]["rewards"]), "prior_max_timestamp": int(values["priors"]["timestamps"].max()),
                "confirmation_queries": len(values["confirmation-queries"]["lengths"]),
                "confirmation_response_read": False, "final_reserve_opened": False,
                "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}}
    write_json(root / "manifest.json", manifest); verify(context)
    return manifest


def load_data(context: dict) -> tuple[dict, dict]:
    root = context["new_data_dir"]
    arrays = {s: load_npz(root / f"{s}.npz") for s in ("small", "expanded", "development", "calibration")}
    features = {s: load_npz(root / f"{s}-features.npz")["features"] for s in arrays}
    return arrays, features
