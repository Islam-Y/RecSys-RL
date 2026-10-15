from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys_rl.data import sha256, write_json
from recsys_rl.history_reward_data import connection, load_npz
from recsys_rl.ranking_data import queries_for
from recsys_rl.recency_data import read_json
from recsys_rl.shortlist_mass_data import load_context as load_mass_context
from recsys_rl.shortlist_mass_data import verify as verify_mass


def verify(context: dict) -> None:
    verify_mass(context)
    for path, expected in context["plan"]["transfer_inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Transfer input changed: {path}")


def load_context(config_path: Path, protocol: Path, output: Path, device: str, resume: bool) -> dict:
    context = load_mass_context(config_path, protocol, output, device, resume)
    config_path = config_path.resolve()
    resolve = lambda key: (config_path.parent / context["config"][key]).resolve()
    report = read_json(resolve("mass_report"))
    execution = read_json(resolve("mass_execution"))
    if sha256(resolve("mass_report")) != execution["result_sha256"]:
        raise ValueError("Fixed mass-rule report changed.")
    for name, expected in report["provenance"]["frozen"]["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical mass-rule source changed: {name}")
    descriptor = report["selection"]["descriptor"]
    rule_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    if descriptor != context["config"]["rule"] or rule_hash != report["selection"]["rule_sha256"]:
        raise ValueError("Transfer rule differs from the previously fixed rule.")
    metadata = read_json(resolve("source_metadata"))
    source = resolve("new_source")
    if source.stat().st_size != metadata["expected_bytes"] or sha256(source) != metadata["expected_sha256"]:
        raise ValueError("New source size or SHA256 differs from pinned metadata.")
    inputs = {str(p): sha256(p) for p in (resolve("mass_report"), resolve("mass_execution"), resolve("source_metadata"))}
    inputs[str(source)] = metadata["expected_sha256"]
    for name, expected in report["provenance"]["frozen"]["inputs"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Historical mass-rule input changed.")
        inputs[name] = expected
    for name, expected in report["provenance"]["extension_inputs"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Historical mass-rule extension changed.")
        inputs[name] = expected
    plan = context["plan"]
    if "transfer_inputs" in plan:
        if plan["transfer_inputs"] != inputs:
            raise ValueError("Transfer plan changed.")
    else:
        plan["transfer_inputs"] = inputs
        plan["fixed_rule_sha256"] = rule_hash
        write_json(context["output"] / "plan.json", plan)
    context.update({"old_source": context["source"], "transfer_source": source, "rule": descriptor, "rule_sha256": rule_hash})
    verify(context)
    return context


def sql_path(path: Path) -> str:
    return "'"+str(path).replace("'", "''")+"'"


def connect(context: dict, source: Path):
    db = connection({**context, "source": source})
    spill = context["output"] / "spill"
    spill.mkdir(exist_ok=True)
    db.execute("SET temp_directory = "+sql_path(spill))
    return db


def compatibility(context: dict, db) -> tuple[np.ndarray, dict]:
    db.read_parquet(str(context["old_source"])).create_view("old_events")
    old = db.execute("SELECT DISTINCT uid FROM old_events WHERE uid IS NOT NULL ORDER BY uid").fetchnumpy()["uid"]
    new = db.execute("SELECT DISTINCT uid FROM raw_events WHERE uid IS NOT NULL ORDER BY uid").fetchnumpy()["uid"]
    common = np.intersect1d(old, new)
    settings = context["config"]["data"]
    if len(common) < settings["compatibility_users"]:
        raise ValueError("Insufficient shared users to establish ID/time compatibility.")
    sample = np.sort(np.random.default_rng(settings["compatibility_seed"]).choice(common, settings["compatibility_users"], replace=False))
    db.register("compatibility_users", {"uid": sample})
    columns = ",".join(f"r.{name}::BIGINT AS {name}" for name in ("uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"))+",r.event_type::VARCHAR AS event_type"
    for name, source in (("old_sample", "old_events"), ("new_sample", "raw_events")):
        db.execute(f"CREATE TEMP TABLE {name} AS SELECT {columns} FROM {source} r JOIN compatibility_users USING(uid)")
    mismatch = db.execute("""SELECT count(*) FROM ((SELECT * FROM old_sample EXCEPT ALL SELECT * FROM new_sample)
        UNION ALL (SELECT * FROM new_sample EXCEPT ALL SELECT * FROM old_sample)) AS differences""").fetchone()[0]
    events = db.execute("SELECT count(*) FROM old_sample").fetchone()[0]
    db.unregister("compatibility_users")
    if mismatch:
        raise ValueError("Source IDs, timestamps or shared event contents differ; do not evaluate.")
    return old, {"old_source_users": len(old), "new_source_users": len(new), "shared_users": len(common),
                 "sample_users": len(sample), "sample_events": events, "multiset_differences": mismatch}


def build(context: dict, id_path: Path) -> tuple[dict, dict]:
    settings = context["config"]["data"]
    if settings["history_length"] != 64 or settings["minimum_history"] != 5:
        raise ValueError("Transfer requires the original history protocol.")
    cuts = context["manifest"]["cutoffs_seconds"]
    db = connect(context, context["transfer_source"])
    try:
        old, check = compatibility(context, db)
        db.register("old_users", {"uid": old})
        db.execute("""CREATE TEMP TABLE inventory AS SELECT uid,
            count(*) FILTER(WHERE event_type='listen' AND timestamp<=?) AS early,
            count(*) FILTER(WHERE event_type='listen') AS total FROM raw_events GROUP BY uid""", [cuts["train_end"]])
        populations = {"old_users": old}
        for name in ("primary", "secondary"):
            spec = settings[name]
            ceiling = "" if spec["maximum_early_events"] is None else " AND early <= "+str(int(spec["maximum_early_events"]))
            selected = db.execute("""SELECT i.uid,early FROM inventory i LEFT JOIN old_users x ON i.uid=x.uid
                WHERE x.uid IS NULL AND total>0 AND early>=?"""+ceiling+" ORDER BY hash(i.uid,?),i.uid LIMIT ?",
                [spec["minimum_early_events"], spec["user_seed"], spec["users"]]).fetchnumpy()
            if len(selected["uid"]) != spec["users"]:
                raise ValueError("Insufficient users for the preregistered transfer cohort.")
            ordering = np.argsort(selected["uid"])
            populations[f"{name}_users"] = selected["uid"][ordering]
            populations[f"{name}_early_counts"] = selected["early"][ordering]
        selected_users = np.union1d(populations["primary_users"], populations["secondary_users"])
        if len(selected_users) != settings["primary"]["users"]+settings["secondary"]["users"] or np.intersect1d(selected_users, old).size:
            raise ValueError("Transfer populations overlap each other or the complete old source.")
        db.register("selected_users", {"uid": selected_users})
        db.execute("""COPY (SELECT r.uid,r.item_id,r.timestamp,'listen'::VARCHAR AS event_type FROM raw_events r
            JOIN selected_users USING(uid) WHERE r.event_type='listen' AND r.timestamp<=? ORDER BY r.uid,r.timestamp,r.item_id)
            TO """+sql_path(id_path)+" (FORMAT PARQUET,COMPRESSION ZSTD)", [cuts["data_end"]])
    finally:
        db.close()
    arrays = {"users": populations}
    profiles = {}
    db = connect(context, id_path)
    try:
        for name in ("primary", "secondary"):
            spec, users = settings[name], populations[f"{name}_users"]
            arrays[name] = queries_for(context, db, users, cuts["validation_end"], cuts["data_end"], spec["query_seed"], limit=spec["queries"])
            db.register("cohort_users", {"uid": users})
            db.register("known_items", {"item_id": np.asarray([i["item_id"] for i in context["catalog"]["items"]], dtype=np.uint32)})
            total, known = db.execute("""SELECT count(*),count(k.item_id) FROM listens l JOIN cohort_users USING(uid)
                LEFT JOIN known_items k USING(item_id) WHERE timestamp>? AND timestamp<=?""", [cuts["validation_end"], cuts["data_end"]]).fetchone()
            db.unregister("cohort_users")
            db.unregister("known_items")
            a = arrays[name]
            profiles[name] = {"selected_users": len(users), "queries": len(a["lengths"]), "query_users": len(np.unique(a["user_ids"])),
                              "targets": len(a["target_tokens"]), "test_listens": total, "test_catalog_listens": known,
                              "test_catalog_coverage": known/total if total else None,
                              "history_unknown_fraction": float(np.count_nonzero(a["input_ids"] == 2)/np.sum(a["lengths"]-1))}
    finally:
        db.close()
    return arrays, {"compatibility": check, "cohorts": profiles}


def prepare(context: dict) -> dict:
    root = context["new_data_dir"]
    if (root / "manifest.json").exists():
        result = read_json(root / "manifest.json")
        if result["plan"] != context["plan"]:
            raise ValueError("Transfer preparation plan changed.")
        for name, expected in result["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Transfer prepared artifact changed.")
        return result
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete transfer preparation.")
    root.mkdir(parents=True, exist_ok=True)
    values, description = build(context, root / "id-events.parquet")
    for name, arrays in values.items():
        np.savez_compressed(root / f"{name}.npz", **arrays)
    result = {**description, "plan": context["plan"], "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "new_response_opened": False, "new_predictions_opened": False,
              "artifacts": {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}}
    write_json(root / "manifest.json", result)
    verify(context)
    return result


def fingerprints(context: dict, source: Path, users: np.ndarray) -> tuple[list, dict]:
    db = connect(context, source)
    try:
        db.register("selected_users", {"uid": users})
        fields = ",".join(f"r.{n}::BIGINT" for n in ("uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"))+",r.event_type::VARCHAR"
        rows = db.execute("SELECT r.uid,count(*),sum(hash("+fields+")),bit_xor(hash("+fields+",42)) FROM raw_events r JOIN selected_users USING(uid) WHERE event_type='listen' AND timestamp<=? GROUP BY r.uid ORDER BY r.uid",
                          [context["manifest"]["cutoffs_seconds"]["data_end"]]).fetchall()
    finally:
        db.close()
    normalized = [[int(v) for v in row] for row in rows]
    digest = hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest()
    return normalized, {"users": len(rows), "rows": sum(int(row[1]) for row in rows), "per_user_fingerprint_sha256": digest}


def extract_responses(context: dict) -> dict:
    root = context["output"] / "history"
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if sha256(root / "events.parquet") != manifest["sha256"]:
            raise ValueError("Extracted response history changed.")
        context["source"] = root / "events.parquet"
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete extracted response history.")
    root.mkdir()
    populations = load_npz(context["new_data_dir"] / "users.npz")
    users = np.union1d(populations["primary_users"], populations["secondary_users"])
    db = connect(context, context["transfer_source"])
    try:
        db.register("selected_users", {"uid": users})
        db.execute("""COPY (SELECT r.* FROM raw_events r JOIN selected_users USING(uid)
            WHERE r.event_type='listen' AND r.timestamp<=? ORDER BY r.uid,r.timestamp,r.item_id)
            TO """+sql_path(root / "events.parquet")+" (FORMAT PARQUET,COMPRESSION ZSTD)", [context["manifest"]["cutoffs_seconds"]["data_end"]])
    finally:
        db.close()
    raw, raw_summary = fingerprints(context, context["transfer_source"], users)
    extracted, extracted_summary = fingerprints(context, root / "events.parquet", users)
    if raw != extracted:
        raise ValueError("Extracted response history differs from the source fingerprints.")
    result = {"source": str(root / "events.parquet"), "sha256": sha256(root / "events.parquet"), "raw": raw_summary,
              "extracted": extracted_summary, "fingerprints_match": True}
    write_json(root / "manifest.json", result)
    context["source"] = root / "events.parquet"
    return result


def query_slice(arrays: dict, start: int, stop: int) -> dict:
    first, last = arrays["target_offsets"][[start, stop]]
    return {**{k: arrays[k][start:stop] for k in ("input_ids", "lengths", "user_ids", "timestamps", "history_max_timestamps")},
            "target_tokens": arrays["target_tokens"][first:last], "target_offsets": arrays["target_offsets"][start:stop+1]-first}
