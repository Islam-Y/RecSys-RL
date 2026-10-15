from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy as np
import torch

from recsys_rl.data import sha256, write_json
from recsys_rl.history_reward_data import attach_observed, history_features, load_npz
from recsys_rl.ranking_data import build, load_context, load_data, prepare, verify
from recsys_rl.ranking_model import accuracies, calibrated, comparisons, emit, evaluate_scores, fit_calibration, predict, quick, train
from recsys_rl.recency_data import read_json
from recsys_rl.reward_evaluation import regression_metrics


def controls(features: np.ndarray) -> dict:
    return {"item_prior": features[:, 8].astype(np.float64), "history_candidate": features[:, 9].astype(np.float64),
            "history_user": features[:, 0].astype(np.float64)}


def choose(context: dict, runs: dict, development_controls: dict) -> dict:
    settings = context["config"]["selection"]
    threshold = max(settings["minimum_query_macro_accuracy"], development_controls["item_prior"], development_controls["history_candidate"])
    initial = [runs[f"{v['name']}/42"] for v in context["config"]["variants"]]
    eligible = [r for r in initial if r["development_query_macro"] >= threshold]
    candidate = max(eligible, key=lambda r: r["development_query_macro"]) if eligible else None
    seeds = [42, *context["config"]["training"]["confirmation_seeds"]]
    confirmation = [{"seed": s, "passed": runs[f"{candidate['variant']['name']}/{s}"]["development_query_macro"] >= threshold}
                    for s in seeds if candidate and f"{candidate['variant']['name']}/{s}" in runs]
    passed = candidate is not None and len(confirmation) == len(seeds) and all(r["passed"] for r in confirmation)
    return {"screening_challenger": candidate["variant"]["name"] if candidate else None,
            "development_threshold": threshold, "confirmation_seeds": confirmation, "accepted_on_development": passed,
            "selected_variant": candidate["variant"]["name"] if passed else None,
            "checkpoint": {"path": candidate["checkpoint"], "sha256": candidate["artifacts"][Path(candidate["checkpoint"]).name]} if passed else None,
            "confirmation_used_for_selection": False, "final_reserve_opened": False}


def eligibility(context: dict, statistics: dict, regression: dict) -> dict:
    settings = context["config"]["selection"]
    value = statistics["gap_010"]["methods"]["selected"]["query_macro"]
    calibrated_value = statistics["gap_010"]["methods"]["calibrated"]["query_macro"]
    paired = statistics["gap_010"]["paired_vs_controls"]["selected"]
    checks = {"enough_informative_queries": statistics["gap_010"]["queries"] >= settings["minimum_confirmation_queries"],
              "query_macro_at_least_055": value is not None and value["estimate"] >= settings["minimum_query_macro_accuracy"],
              "query_macro_ci_above_chance": value is not None and value["ci95"][0] > .5,
              "paired_improvement_over_item_prior": paired["item_prior"] is not None and paired["item_prior"]["ci95"][0] > 0,
              "paired_improvement_over_history_candidate": paired["history_candidate"] is not None and paired["history_candidate"]["ci95"][0] > 0,
              "calibrated_mse_preserved": regression["calibrated"]["mse"] <= settings["maximum_calibrated_mse_ratio"]*regression["history_user"]["mse"],
              "calibrated_query_macro_preserved": calibrated_value is not None and calibrated_value["estimate"] >= settings["minimum_query_macro_accuracy"]}
    return {"checks": checks, "rl_allowed": all(checks.values()), "final_reserve_opened": False}


def open_confirmation(context: dict, selected: dict, runs: dict, calibration: dict) -> dict:
    root = context["output"] / "confirmation"
    if (root / "result.json").exists(): return read_json(root / "result.json")
    if root.exists() and any(root.iterdir()): raise FileExistsError("Preserve incomplete ranking confirmation.")
    root.mkdir()
    opening = {"opened_at_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": sha256(context["output"] / "selection.json"),
               "calibration_sha256": sha256(context["output"] / "calibration.json")}
    write_json(root / "opening.json", opening); emit(stage="ranking_confirmation_opening", selected=selected["selected_variant"])
    queries = load_npz(context["new_data_dir"] / "confirmation-queries.npz")
    arrays = attach_observed(context, queries)
    feature_context = {**context, "train": load_npz(context["new_data_dir"] / "priors.npz")}
    features = history_features(feature_context, arrays, False)
    run = runs[f"{selected['selected_variant']}/42"]
    scores = controls(features["features"])
    scores["selected"] = predict(context, run, features["features"])
    scores["calibrated"] = calibrated(scores["selected"], features["features"], calibration)
    other_data = "expanded" if run["variant"]["data"] == "small" else "small"
    other = next(v for v in context["config"]["variants"] if v["model"] == run["variant"]["model"] and v["data"] == other_data)
    scores["matched_data_control"] = predict(context, runs[f"{other['name']}/42"], features["features"])
    statistics = evaluate_scores(context, arrays, scores)
    regression = {n: regression_metrics(arrays["rewards"], scores[n]) for n in ("calibrated", "history_user", "history_candidate", "item_prior")}
    decision = eligibility(context, statistics, regression)
    for name, values in (("observed", arrays), ("features", features), ("scores", scores)):
        np.savez_compressed(root / f"{name}.npz", **values)
    result = {"opening": opening, "statistics": statistics, "regression": regression, "decision": decision,
              "matched_data_control": other, "calibration": calibration,
              "artifacts": {p.name: sha256(p) for p in (root / "opening.json", root / "observed.npz", root / "features.npz", root / "scores.npz")},
              "response_labels_read": True, "final_reserve_opened": False}
    write_json(root / "result.json", result)
    return result


def run_series(context: dict, preparation: dict) -> dict:
    root = context["output"]
    if (root / "result.json").exists(): return read_json(root / "result.json")
    started = time.perf_counter()
    arrays, features = load_data(context)
    index = comparisons(context, arrays["development"])
    control_scores = controls(features["development"])
    control_means = {n: quick(context, arrays["development"], index, v) for n, v in control_scores.items()}
    runs = {}
    for variant in context["config"]["variants"]:
        runs[f"{variant['name']}/42"] = train(context, arrays, features, variant, 42)
    selected = choose(context, runs, control_means)
    if selected["screening_challenger"]:
        variant = next(v for v in context["config"]["variants"] if v["name"] == selected["screening_challenger"])
        for seed in context["config"]["training"]["confirmation_seeds"]:
            runs[f"{variant['name']}/{seed}"] = train(context, arrays, features, variant, seed)
    selected = choose(context, runs, control_means)
    selected["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "selection.json", selected)
    development_scores = dict(control_scores)
    for key, record in runs.items(): development_scores[key] = np.load(Path(record["checkpoint"]).parent / "development.npy", allow_pickle=False)
    development = evaluate_scores(context, arrays["development"], development_scores)
    np.savez_compressed(root / "development-scores.npz", **development_scores)
    calibration, confirmation = None, None
    if selected["accepted_on_development"]:
        record = runs[f"{selected['selected_variant']}/42"]
        scores = predict(context, record, features["calibration"])
        calibration = fit_calibration(context, scores, features["calibration"], arrays["calibration"]["rewards"])
        np.save(root / "calibration-scores.npy", scores, allow_pickle=False)
        write_json(root / "calibration.json", {**calibration, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                                               "checkpoint_sha256": selected["checkpoint"]["sha256"]})
        confirmation = open_confirmation(context, selected, runs, calibration)
    verify(context)
    artifact_names = ["selection.json", "development-scores.npz"]
    if calibration: artifact_names.extend(["calibration.json", "calibration-scores.npy"])
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "preparation": preparation,
              "runs": runs, "selection": selected, "development": development,
              "calibration": calibration, "confirmation": confirmation,
              "rl_allowed": confirmation["decision"]["rl_allowed"] if confirmation else False,
              "rl_trained": False, "final_reserve_opened": False, "seconds": time.perf_counter()-started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "catboost": catboost.__version__, "device": "cpu"},
              "artifacts": {name: sha256(root / name) for name in artifact_names},
              "limitations": ["Pair evaluation conditional on observed same-user/same-timestamp multi-target groups, not arbitrary recommendations.",
                              "Larger training data and ranking-specific selection differ from previous experiments; no single-factor historical attribution.",
                              "Expanded development remains adaptive; confirmation uses new users, same source/time/catalog."]}
    write_json(root / "result.json", result)
    emit(stage="ranking_complete", runs=len(runs), selected=selected["selected_variant"], rl_allowed=result["rl_allowed"], final_reserve_opened=False)
    return result


def audit(context: dict, preparation: dict) -> dict:
    root = context["output"]; result = read_json(root / "result.json")
    for name, expected in result["artifacts"].items():
        if sha256(root / name) != expected: raise ValueError("Ranking result artifact changed.")
    rebuilt = build(context)
    for name, arrays in rebuilt.items():
        saved = load_npz(context["new_data_dir"] / f"{name}.npz")
        if saved.keys() != arrays.keys() or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise ValueError(f"Ranking raw preparation mismatch: {name}")
    arrays, features = load_data(context)
    development_scores = controls(features["development"]); maximum = 0.0
    for key, record in result["runs"].items():
        path = Path(record["checkpoint"]).parent
        for name, expected in record["artifacts"].items():
            if sha256(path / name) != expected: raise ValueError("Ranking run artifact changed.")
        scores = predict(context, record, features["development"])
        saved = np.load(path / "development.npy", allow_pickle=False)
        difference = float(np.max(np.abs(scores-saved))); maximum = max(maximum, difference)
        if difference > 1e-8: raise ValueError("Ranking forward mismatch.")
        index = comparisons(context, arrays["development"])
        if quick(context, arrays["development"], index, scores) != record["development_query_macro"]:
            raise ValueError("Ranking development metric mismatch.")
        history = read_json(path / "training-history.json")["checkpoints"]
        if history != record["history"] or max(history, key=lambda r: r["development_query_macro"])["checkpoint_step"] != record["selected_step"]:
            raise ValueError("Ranking checkpoint selection mismatch.")
        development_scores[key] = scores
        emit(stage="ranking_audit_forward", run=key, maximum_difference=difference)
    development = evaluate_scores(context, arrays["development"], development_scores)
    if development != result["development"]: raise ValueError("Ranking development evaluation mismatch.")
    index = comparisons(context, arrays["development"])
    means = {n: quick(context, arrays["development"], index, v) for n, v in controls(features["development"]).items()}
    selected = choose(context, result["runs"], means)
    if selected != {k: v for k, v in result["selection"].items() if k != "recorded_at_utc"}:
        raise ValueError("Ranking recipe selection mismatch.")
    if result["confirmation"]:
        run = result["runs"][f"{selected['selected_variant']}/42"]
        calibration_scores = predict(context, run, features["calibration"])
        calibration = fit_calibration(context, calibration_scores, features["calibration"], arrays["calibration"]["rewards"])
        if calibration != result["calibration"]: raise ValueError("Ranking calibration mismatch.")
        location = root / "confirmation"; report = read_json(location / "result.json")
        if report != result["confirmation"]: raise ValueError("Confirmation report changed.")
        for name, expected in report["artifacts"].items():
            if sha256(location / name) != expected: raise ValueError("Confirmation evidence changed.")
        calibration_record = read_json(root / "calibration.json")
        opening = report["opening"]
        if opening["opened_at_utc"] <= calibration_record["recorded_at_utc"] or opening["opened_at_utc"] <= result["selection"]["recorded_at_utc"] or opening["selection_sha256"] != sha256(root / "selection.json") or opening["calibration_sha256"] != sha256(root / "calibration.json"):
            raise ValueError("Ranking confirmation opened before model/calibration freeze.")
        observed = attach_observed(context, rebuilt["confirmation-queries"])
        saved = load_npz(location / "observed.npz")
        if any(not np.array_equal(saved[k], v) for k, v in observed.items()): raise ValueError("Confirmation label join mismatch.")
        feature_context = {**context, "train": rebuilt["priors"]}
        fresh = history_features(feature_context, observed, False)
        saved_features = load_npz(location / "features.npz")
        if any(not np.array_equal(saved_features[k], v) for k, v in fresh.items()): raise ValueError("Confirmation past-response features mismatch.")
        scores = controls(fresh["features"]); scores["selected"] = predict(context, run, fresh["features"])
        scores["calibrated"] = calibrated(scores["selected"], fresh["features"], calibration)
        counterpart = report["matched_data_control"]["name"]
        scores["matched_data_control"] = predict(context, result["runs"][f"{counterpart}/42"], fresh["features"])
        stored = load_npz(location / "scores.npz")
        if any(not np.allclose(stored[k], v, rtol=0, atol=1e-8) for k, v in scores.items()): raise ValueError("Confirmation ranking forward mismatch.")
        statistics = evaluate_scores(context, observed, scores)
        regression = {n: regression_metrics(observed["rewards"], scores[n]) for n in ("calibrated", "history_user", "history_candidate", "item_prior")}
        if statistics != report["statistics"] or regression != report["regression"] or eligibility(context, statistics, regression) != report["decision"]:
            raise ValueError("Confirmation metrics/decision mismatch.")
    verify(context)
    return {"stage": "ranking_audit_complete", "runs_checked": len(result["runs"]), "maximum_forward_difference": maximum,
            "raw_data_priors_histories_reconstructed": True, "selection_and_metrics_recalculated": True,
            "confirmation_recalculated": result["confirmation"] is not None, "rl_allowed": result["rl_allowed"], "final_reserve_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare neural and boosted candidate rankers on expanded observed training data.")
    parser.add_argument("--config", type=Path, default=Path("configs/ranking.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/ranking-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/candidate-ranking-v1"))
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config, args.protocol, args.output_dir, args.resume or args.audit)
    preparation = prepare(context)
    if args.prepare_only: emit(stage="ranking_prepared", partitions=preparation["partitions"], populations=preparation["populations"], confirmation_response_read=False, final_reserve_opened=False)
    elif args.audit: emit(**audit(context, preparation))
    else: run_series(context, preparation)


if __name__ == "__main__": main()
