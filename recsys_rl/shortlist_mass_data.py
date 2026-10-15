from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys_rl.candidate_policy_data import load_context as load_base_context
from recsys_rl.candidate_policy_data import verify as verify_base
from recsys_rl.data import sha256, write_json
from recsys_rl.history_reward_data import connection, load_npz
from recsys_rl.ranking_data import queries_for
from recsys_rl.recency_data import read_json


def verify(context: dict) -> None:
    verify_base(context)
    for path, expected in context["plan"]["extension_inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Shortlist-mass input changed: {path}")


def load_context(config_path: Path, protocol: Path, output: Path, device: str, resume: bool) -> dict:
    context = load_base_context(config_path, protocol, output, device, resume)
    config_path = config_path.resolve()
    resolve = lambda key: (config_path.parent / context["config"][key]).resolve()
    previous = read_json(resolve("previous_report"))
    execution = read_json(resolve("previous_execution"))
    if sha256(resolve("previous_report")) != execution["result_sha256"]:
        raise ValueError("Previous candidate-policy report changed.")
    for name, expected in previous["provenance"]["frozen"]["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical candidate computation changed: {name}")
    data_dir, cache_dir = resolve("development_data_dir"), resolve("development_output_dir") / "cache"
    extension = {str(p): sha256(p) for p in (resolve("previous_report"), resolve("previous_execution"), data_dir / "manifest.json", cache_dir / "manifest.json")}
    for name, expected in previous["provenance"]["frozen"]["inputs"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Historical candidate-policy input changed.")
        extension[name] = expected
    for root, hashes in ((data_dir, previous["preparation"]["artifacts"]), (cache_dir, previous["cache"]["artifacts"])):
        for name, expected in hashes.items():
            if sha256(root / name) != expected:
                raise ValueError("Previous candidate-policy artifact changed.")
            extension[str(root / name)] = expected
    selected = previous["selection"]["selected_rl"]
    if selected is None:
        raise ValueError("Expected a fixed expanded RL comparator.")
    expanded = previous["selection"]["selected_descriptors"][selected]
    path = Path(expanded["checkpoint"])
    if sha256(path) != expanded["checkpoint_sha256"]:
        raise ValueError("Expanded RL comparator changed.")
    extension[str(path)] = expanded["checkpoint_sha256"]
    plan = context["plan"]
    if "extension_inputs" in plan:
        if plan["extension_inputs"] != extension:
            raise ValueError("Shortlist-mass extension plan changed.")
    else:
        plan["extension_inputs"] = extension
        write_json(context["output"] / "plan.json", plan)
    context.update({"validation": load_npz(data_dir / "validation.npz"),
                    "development_cache": load_npz(cache_dir / "validation.npz"),
                    "previous_groups": load_npz(data_dir / "populations.npz"),
                    "expanded_descriptor": expanded, "previous_report": previous})
    verify(context)
    return context


def build_final(context: dict) -> dict:
    settings = context["config"]["data"]
    excluded = np.unique(np.concatenate(list(context["previous_groups"].values())))
    if len(excluded) != settings["expected_excluded_users"]:
        raise ValueError("Unexpected historical population size.")
    db = connection(context)
    try:
        db.register("excluded_users", {"uid": excluded})
        users = db.execute("""SELECT DISTINCT l.uid FROM listens l LEFT JOIN excluded_users x ON l.uid=x.uid
            WHERE x.uid IS NULL ORDER BY l.uid""").fetchnumpy()["uid"]
        db.unregister("excluded_users")
        if len(users) != settings["expected_final_users"] or np.intersect1d(users, excluded).size:
            raise ValueError("Unused final population differs from preregistration.")
        db.register("remaining_users", {"uid": users})
        cuts = context["manifest"]["cutoffs_seconds"]
        early = db.execute("""SELECT uid,count(*) AS listens FROM listens JOIN remaining_users USING(uid)
            WHERE timestamp<=? GROUP BY uid""", [cuts["train_end"]]).fetchnumpy()
        db.unregister("remaining_users")
        lookup = {int(u): int(n) for u, n in zip(early["uid"], early["listens"])}
        counts = np.asarray([lookup.get(int(u), 0) for u in users], dtype=np.int64)
        queries = queries_for(context, db, users, cuts["validation_end"], cuts["data_end"], settings["query_seed"], limit=settings["final_queries"])
    finally:
        db.close()
    return {"final": queries, "users": {"users": users, "excluded_users": excluded, "early_counts": counts}}


def prepare(context: dict) -> dict:
    root = context["new_data_dir"]
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if manifest["frozen"] != context["plan"]["frozen"] or manifest["extension_inputs"] != context["plan"]["extension_inputs"]:
            raise ValueError("Shortlist-mass preparation changed.")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Shortlist-mass prepared artifact changed.")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Preserve incomplete shortlist-mass preparation.")
        values = build_final(context)
        root.mkdir(parents=True, exist_ok=True)
        for name, arrays in values.items():
            np.savez_compressed(root / f"{name}.npz", **arrays)
        arrays, users = values["final"], values["users"]
        counts = users["early_counts"]
        manifest = {"frozen": context["plan"]["frozen"], "extension_inputs": context["plan"]["extension_inputs"],
                    "source_commit": context["plan"]["source_commit"], "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "selected_users": len(users["users"]), "excluded_users": len(users["excluded_users"]),
                    "early_activity_populations": {"0-4": int(np.count_nonzero(counts < 5)), "5-49": int(np.count_nonzero((counts >= 5) & (counts < 50))), "50+": int(np.count_nonzero(counts >= 50))},
                    "queries": len(arrays["lengths"]), "query_users": len(np.unique(arrays["user_ids"])), "targets": len(arrays["target_tokens"]),
                    "scope": "All remaining users, including less active and late-arriving users; differs from previous early-active cohorts.",
                    "final_response_read": False, "final_predictions_read": False,
                    "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}}
        write_json(root / "manifest.json", manifest)
    verify(context)
    return manifest
