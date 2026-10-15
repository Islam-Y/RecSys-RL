from __future__ import annotations

import argparse
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys_rl.data import sha256, write_json
from recsys_rl.history_reward_data import load_npz
from recsys_rl.ranking_model import emit
from recsys_rl.recency_data import read_json
from recsys_rl.transfer_data import build, extract_responses, fingerprints, load_context, prepare, verify
from recsys_rl.transfer_scoring import cohort_metrics, collect_chunks, models, rollout_records


COHORTS = ("primary", "secondary")


def check_arrays(actual: dict, path: Path) -> None:
    saved = load_npz(path)
    if set(actual) != set(saved) or any(not np.array_equal(value, saved[name]) for name, value in actual.items()):
        raise ValueError(f"Transfer arrays differ: {path}")


def opening(context: dict, preparation: dict, audit: bool = False) -> dict:
    path = context["output"] / "opening.json"
    expected = {"rule": context["rule"], "rule_sha256": context["rule_sha256"],
                "preparation_sha256": sha256(context["new_data_dir"] / "manifest.json"),
                "source_sha256": context["plan"]["transfer_inputs"][str(context["transfer_source"])],
                "checkpoints": {str(p): sha256(Path(p)) for p in (context["policy_checkpoint"], context["historical_checkpoint"], context["expanded_descriptor"]["checkpoint"], context["ranker_run"]["checkpoint"])},
                "tuning_allowed": False}
    if path.exists():
        value = read_json(path)
        if expected != {k: v for k, v in value.items() if k != "opened_at_utc"}:
            raise ValueError("Transfer opening differs from fixed inputs.")
    else:
        if audit:
            raise FileNotFoundError("Transfer opening is missing.")
        value = {**expected, "opened_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(path, value)
        emit(stage="transfer_opening", rule_sha256=context["rule_sha256"], cohorts=list(COHORTS))
    if value["opened_at_utc"] <= preparation["recorded_at_utc"]:
        raise ValueError("Transfer responses opened before preparation.")
    return value


def cohort_run(context: dict, name: str, policies: dict) -> dict:
    root = context["output"] / name
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    root.mkdir(exist_ok=True)
    started = time.perf_counter()
    arrays = load_npz(context["new_data_dir"] / f"{name}.npz")
    metadata, rankings, vectors, predictions, inputs, chunks, _ = collect_chunks(context, name, arrays, policies, False)
    for stem, values in (("rankings", rankings), ("vectors", vectors), ("predictions", predictions), ("rollout-inputs", inputs)):
        np.savez_compressed(root / f"{stem}.npz", **values)
    records = rollout_records(context, name, arrays, inputs, policies["reference"], False)
    metrics = cohort_metrics(context, arrays, metadata, vectors, records)
    result = {**metrics, "cohort": name, "role": "primary_confirmation" if name == "primary" else "secondary_descriptive",
              "queries": len(arrays["lengths"]), "users": len(np.unique(arrays["user_ids"])), "observed_pairs": len(predictions["labels"]),
              "chunks": chunks, "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))},
              "seconds": time.perf_counter()-started, "used_for_selection": False, "causal_user_effect_measured": False}
    write_json(root / "result.json", result)
    emit(stage="transfer_cohort_complete", cohort=name, queries=result["queries"], users=result["users"], confirmation=result["confirmation"])
    return result


def run_series(context: dict, preparation: dict) -> dict:
    root = context["output"]
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    started = time.perf_counter()
    opened = opening(context, preparation)
    history = extract_responses(context)
    policies = models(context)
    cohorts = {name: cohort_run(context, name, policies) for name in COHORTS}
    verify(context)
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "preparation": preparation,
              "opening": opened, "history": history, "cohorts": cohorts,
              "primary_confirmation": cohorts["primary"]["confirmation"],
              "series_seconds": time.perf_counter()-started, "new_training_runs": 0, "tuning_performed": False,
              "experimental_work_complete": True, "pdf_changed": False,
              "scope": "New users excluded from the complete original 50M source; same service and period, fixed 2000-item catalog. Secondary does not replace primary."}
    write_json(root / "result.json", result)
    emit(stage="transfer_complete", primary=result["primary_confirmation"], cohorts=len(cohorts), new_training_runs=0)
    return result


def audit(context: dict, preparation: dict) -> dict:
    root = context["output"]
    result = read_json(root / "result.json")
    if result["provenance"] != context["plan"] or result["preparation"] != preparation or result["opening"] != opening(context, preparation, True):
        raise ValueError("Transfer report provenance or opening differs.")
    with tempfile.TemporaryDirectory(prefix="transfer-audit-", dir=root) as temporary:
        rebuilt, description = build(context, Path(temporary) / "id-events.parquet")
        for name, arrays in rebuilt.items():
            check_arrays(arrays, context["new_data_dir"] / f"{name}.npz")
        if any(description[name] != preparation[name] for name in description):
            raise ValueError("Transfer raw preparation description differs.")
    history = extract_responses(context)
    users = np.union1d(rebuilt["users"]["primary_users"], rebuilt["users"]["secondary_users"])
    raw, raw_summary = fingerprints(context, context["transfer_source"], users)
    extracted, extracted_summary = fingerprints(context, context["source"], users)
    if history != result["history"] or raw != extracted or raw_summary != history["raw"] or extracted_summary != history["extracted"]:
        raise ValueError("Transfer response source fingerprints differ.")
    policies = models(context)
    maximum, chunk_count, rollout_count = 0.0, 0, 0
    for name in COHORTS:
        location = root / name
        cohort = read_json(location / "result.json")
        if cohort != result["cohorts"][name]:
            raise ValueError("Transfer cohort result differs.")
        for artifact, expected in cohort["artifacts"].items():
            if sha256(location / artifact) != expected:
                raise ValueError("Transfer aggregate artifact changed.")
        for entry in cohort["chunks"]:
            if sha256(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("Transfer chunk summary changed.")
        arrays = rebuilt[name]
        metadata, rankings, vectors, predictions, inputs, chunks, difference = collect_chunks(context, name, arrays, policies, True)
        maximum = max(maximum, difference)
        chunk_count += len(chunks)
        if chunks != cohort["chunks"]:
            raise ValueError("Transfer chunk inventory differs.")
        for stem, values in (("rankings", rankings), ("vectors", vectors), ("predictions", predictions), ("rollout-inputs", inputs)):
            check_arrays(values, location / f"{stem}.npz")
        records = rollout_records(context, name, arrays, inputs, policies["reference"], True)
        rollout_count += sum(len(modes) for modes in records.values())
        metrics = cohort_metrics(context, arrays, metadata, vectors, records)
        if any(value != cohort[key] for key, value in metrics.items()):
            raise ValueError("Transfer aggregate metrics or confirmation differ.")
        if cohort["queries"] != len(arrays["lengths"]) or cohort["users"] != len(np.unique(arrays["user_ids"])) or cohort["observed_pairs"] != len(predictions["labels"]):
            raise ValueError("Transfer aggregate sample size differs.")
    if result["primary_confirmation"] != result["cohorts"]["primary"]["confirmation"] or result["new_training_runs"] != 0 or result["tuning_performed"]:
        raise ValueError("Transfer primary decision or no-tuning contract differs.")
    verify(context)
    return {"stage": "transfer_audit_complete", "cohorts_checked": len(COHORTS), "chunks_checked": chunk_count,
            "raw_preparation_reproduced": True, "response_fingerprints_match": True,
            "maximum_forward_difference": maximum, "mass_and_identity_checks_passed": True,
            "rollouts_replayed": rollout_count, "decisions_recalculated": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/transfer.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/transfer-protocol.md"))
    parser.add_argument("--output", type=Path, default=Path("outputs/transfer-v1"))
    parser.add_argument("--device", default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config, args.protocol, args.output, args.device, args.resume or args.audit)
    preparation = prepare(context)
    if args.prepare_only:
        emit(stage="transfer_prepared", compatibility=preparation["compatibility"], cohorts=preparation["cohorts"], responses_opened=False)
    elif args.audit:
        emit(**audit(context, preparation))
    else:
        run_series(context, preparation)


if __name__ == "__main__":
    main()
