from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

PAD, BOS, UNK, FIRST_ITEM = 0, 1, 2, 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    data = config["data"]
    if not 0 < data["train_quantile"] < data["validation_quantile"] < 1:
        raise ValueError("Temporal quantiles must satisfy 0 < train < validation < 1.")
    if not 1 <= data["minimum_history"] <= data["history_length"]:
        raise ValueError("Minimum history must fit in the context.")
    if min(data["max_users"], data["catalog_size"], *data["max_queries"].values()) < 1:
        raise ValueError("User, catalog and query budgets must be positive.")
    model = config["model"]
    if min(model["embedding_size"], model["layers"], model["heads"]) < 1:
        raise ValueError("Model dimensions must be positive.")
    if model["embedding_size"] % model["heads"]:
        raise ValueError("Embedding size must be divisible by the number of heads.")
    top_k = config["evaluation"]["top_k"]
    if 10 not in top_k or len(set(top_k)) != len(top_k) or min(top_k) < 1:
        raise ValueError("Ranking cutoffs must be positive, unique and include 10 for diagnostics.")
    if config["evaluation"]["checkpoint_metric"] not in {f"ndcg@{k}" for k in top_k}:
        raise ValueError("Checkpoint metric must be an evaluated NDCG cutoff.")
    training = config["training"]
    if min(training["epochs"], training["batch_size"], training["learning_rate"], training["gradient_clip"]) <= 0:
        raise ValueError("Training budget, learning rate and gradient clip must be positive.")
    return config


def prepare(source: Path, output: Path, config_path: Path, protocol_path: Path) -> dict:
    source = source.resolve(strict=True)
    config = load_config(config_path)
    settings = config["data"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preparation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '4GB'")
    try:
        connection.read_parquet(str(source)).create_view("raw_events")
        columns = {row[0] for row in connection.execute("DESCRIBE raw_events").fetchall()}
        required = {"uid", "item_id", "timestamp", "event_type", "played_ratio_pct", "track_length_seconds"}
        if not required <= columns:
            raise ValueError(f"Missing Yambda columns: {sorted(required - columns)}")
        connection.execute("""CREATE VIEW all_listens AS SELECT * FROM raw_events
            WHERE event_type = 'listen' AND uid IS NOT NULL AND item_id IS NOT NULL AND timestamp IS NOT NULL""")
        total, users, items, first, last, train_end, validation_end = connection.execute("""
            SELECT COUNT(*), COUNT(DISTINCT uid), COUNT(DISTINCT item_id), MIN(timestamp), MAX(timestamp),
                   quantile_disc(timestamp, ?), quantile_disc(timestamp, ?)
            FROM all_listens
        """, [settings["train_quantile"], settings["validation_quantile"]]).fetchone()
        if not first < train_end < validation_end < last:
            raise ValueError("Dataset cannot be split into three temporal periods.")
        connection.execute("""CREATE TEMP TABLE population AS
            SELECT uid FROM all_listens WHERE timestamp <= ? GROUP BY uid HAVING COUNT(*) >= ?
            ORDER BY hash(uid, ?), uid LIMIT ?
        """, [train_end, settings["minimum_train_events"], config["seed"], settings["max_users"]])
        event_path = (output / "listen-events.parquet").resolve()
        connection.sql("""
            SELECT uid AS user_id, item_id, timestamp, played_ratio_pct, track_length_seconds
            FROM all_listens JOIN population USING (uid) ORDER BY uid, timestamp, item_id
        """).write_parquet(str(event_path), compression="zstd")
        connection.read_parquet(str(event_path)).create_view("events")
        connection.execute("""CREATE TEMP TABLE catalog AS
            SELECT item_id, COUNT(*) AS frequency,
                   median(track_length_seconds) FILTER (WHERE track_length_seconds > 0) AS duration_seconds
            FROM events WHERE timestamp <= ? GROUP BY item_id
            ORDER BY frequency DESC, item_id LIMIT ?
        """, [train_end, settings["catalog_size"]])
        catalog_rows = connection.execute("SELECT * FROM catalog ORDER BY frequency DESC, item_id").fetchall()
        if len(catalog_rows) < max(config["evaluation"]["top_k"]):
            raise ValueError("Catalog is smaller than the requested ranking cutoff.")
        catalog = {"special_tokens": {"PAD": PAD, "BOS": BOS, "UNK": UNK}, "items": [
            {"token": index + FIRST_ITEM, "item_id": int(item), "train_frequency": int(frequency),
             "duration_seconds": float(duration) if duration is not None else None}
            for index, (item, frequency, duration) in enumerate(catalog_rows)
        ]}
        write_json(output / "catalog.json", catalog)
        periods = {"train": (-1, train_end), "validation": (train_end, validation_end), "test": (validation_end, last)}
        query_arrays = {}
        coverage = {}
        for split, (lower, upper) in periods.items():
            query_arrays[split] = connection.execute("""
                SELECT user_id, timestamp FROM (
                    SELECT DISTINCT user_id, timestamp FROM events JOIN catalog USING (item_id)
                    WHERE timestamp > ? AND timestamp <= ?
                ) AS candidates
                ORDER BY hash(user_id, timestamp, ?), user_id, timestamp LIMIT ?
            """, [lower, upper, config["seed"], settings["max_queries"][split]]).fetchnumpy()
            row = connection.execute("""
                SELECT COUNT(*), COUNT(*) FILTER (WHERE catalog.item_id IS NOT NULL)
                FROM events LEFT JOIN catalog USING (item_id) WHERE timestamp > ? AND timestamp <= ?
            """, [lower, upper]).fetchone()
            coverage[split] = {"listen_events": row[0], "in_catalog_events": row[1], "event_coverage": row[1] / row[0] if row[0] else 0.0}
        events = connection.execute("SELECT user_id, timestamp, item_id FROM events ORDER BY user_id, timestamp, item_id").fetchnumpy()
        population = int(connection.execute("SELECT COUNT(*) FROM population").fetchone()[0])
    finally:
        connection.close()
    print(json.dumps({"stage": "source_prepared", "users": population, "catalog": len(catalog_rows), "listen_events": len(events["user_id"])}), flush=True)
    item_ids = np.asarray([row[0] for row in catalog_rows], dtype=np.uint32)
    ordered = np.argsort(item_ids)
    sorted_ids = item_ids[ordered]
    indices = np.searchsorted(sorted_ids, events["item_id"])
    in_catalog = (indices < len(sorted_ids)) & (sorted_ids[np.minimum(indices, len(sorted_ids) - 1)] == events["item_id"])
    tokens = np.full(len(indices), UNK, dtype=np.int32)
    tokens[in_catalog] = ordered[indices[in_catalog]] + FIRST_ITEM
    user_ids, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(user): (int(start), int(start + count)) for user, start, count in zip(user_ids, starts, counts)}
    summaries = {}
    for split, queries in query_arrays.items():
        order = np.lexsort((queries["timestamp"], queries["user_id"]))
        inputs, lengths, flat_targets, offsets, users_out, times, history_times = [], [], [], [0], [], [], []
        skipped = 0
        multiple_targets = 0
        for index in order:
            user, timestamp = int(queries["user_id"][index]), int(queries["timestamp"][index])
            start, end = boundaries[user]
            user_times = events["timestamp"][start:end]
            stop = start + int(np.searchsorted(user_times, timestamp, side="left"))
            target_stop = start + int(np.searchsorted(user_times, timestamp, side="right"))
            if stop - start < settings["minimum_history"]:
                skipped += 1
                continue
            target = np.unique(tokens[stop:target_stop])
            target = target[target >= FIRST_ITEM]
            if not len(target):
                raise ValueError("Sampled query has no in-catalog target.")
            history = tokens[max(start, stop - settings["history_length"]):stop]
            previous_time = int(events["timestamp"][stop - 1])
            if previous_time >= timestamp:
                raise ValueError("Point-in-time violation: context contains a target-time event.")
            row = np.zeros(settings["history_length"] + 1, dtype=np.int32)
            row[0] = BOS
            row[1:len(history) + 1] = history
            inputs.append(row)
            lengths.append(len(history) + 1)
            flat_targets.extend(target.tolist())
            offsets.append(len(flat_targets))
            users_out.append(user)
            times.append(timestamp)
            history_times.append(previous_time)
            multiple_targets += len(target) > 1
        if not inputs:
            raise ValueError(f"Empty partition: {split}")
        path = output / f"{split}.npz"
        np.savez_compressed(path, input_ids=np.asarray(inputs, dtype=np.int32), lengths=np.asarray(lengths, dtype=np.int16),
            target_tokens=np.asarray(flat_targets, dtype=np.int32), target_offsets=np.asarray(offsets, dtype=np.int64),
            user_ids=np.asarray(users_out, dtype=np.uint32), timestamps=np.asarray(times, dtype=np.uint32),
            history_max_timestamps=np.asarray(history_times, dtype=np.uint32))
        summaries[split] = {"queries": len(inputs), "users": len(set(users_out)), "targets": len(flat_targets),
            "multi_target_queries": int(multiple_targets), "skipped_short_histories": skipped, "sha256": sha256(path), **coverage[split]}
        print(json.dumps({"stage": "partition_prepared", "split": split, "queries": len(inputs)}), flush=True)
    manifest = {"source": {"dataset": "yandex/yambda", "file": "flat/50m/multi_event.parquet", "sha256": sha256(source), "bytes": source.stat().st_size,
        "listen_events": total, "users": users, "items": items, "timestamp_min": first, "timestamp_max": last},
        "cutoffs_seconds": {"train_end": train_end, "validation_end": validation_end, "data_end": last},
        "config": config, "config_sha256": sha256(config_path), "protocol_sha256": sha256(protocol_path),
        "population_users": population, "catalog_size": len(catalog_rows), "catalog_sha256": sha256(output / "catalog.json"),
        "partitions": summaries, "duckdb": duckdb.__version__, "numpy": np.__version__, "test_evaluated": False,
        "ranking_scope": "Full train-selected restricted catalog; conditional on an in-catalog observed next-listen target."}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_partition(directory: Path, split: str) -> dict[str, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}.npz"
    if sha256(path) != manifest["partitions"][split]["sha256"]:
        raise ValueError(f"Prepared partition hash differs: {split}")
    with np.load(path, allow_pickle=False) as file:
        arrays = dict(file)
    if not np.all(arrays["history_max_timestamps"] < arrays["timestamps"]):
        raise ValueError("Point-in-time violation in saved sequences.")
    return arrays
