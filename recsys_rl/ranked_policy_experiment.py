from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostRanker

from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings
from recsys_rl.history_reward_data import attach_observed, history_features, load_npz
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.ranked_policy_data import catalog_reward, load_context, mixture, prepare_cache, verify
from recsys_rl.ranked_policy_training import choose, evaluate, hard_repeat, intervals, rollout, train
from recsys_rl.ranking_model import calibrated, emit
from recsys_rl.recency_data import read_json
from recsys_rl.reward_evaluation import regression_metrics


def validation_controls(context: dict, caches: dict) -> tuple[dict, dict]:
    arrays = context["validation"]; cache = caches["validation"]
    reference = mixture(context, arrays, cache["reference_logits"])
    controls, vectors = {}, {}
    for name, p in (("reference", reference), ("hard-repeat", hard_repeat(arrays, reference))):
        controls[name], vectors[name], _ = evaluate(context, arrays, cache["reward"], p, reference=reference)
    for name, ranking in control_rankings(arrays, context["catalog"], 20).items():
        controls[name], vectors[name], _ = evaluate(context, arrays, cache["reward"], rankings=ranking)
    for variant in context["config"]["variants"]:
        r = cache["reward"].copy(); last = arrays["input_ids"][np.arange(len(r)), arrays["lengths"]-1]; valid = last >= 3
        r[np.flatnonzero(valid), last[valid]-3] -= variant["repeat_penalty"]
        logits = np.log(np.maximum(reference, 1e-30))+r/variant["beta"]
        p = np.exp(logits-logits.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        controls[f"analytic-{variant['name']}"], _, _ = evaluate(context, arrays, cache["reward"], p, reference=reference)
    return controls, vectors


def save_rollouts(context: dict, caches: dict, key: str, record: dict | None, hard: bool = False) -> dict:
    root = context["output"] / "rollouts"; root.mkdir(exist_ok=True)
    output = {}
    for mode in context["config"]["rollout"]["modes"]:
        saved, summary = rollout(context, caches["validation"], record, mode, hard)
        path = root / f"{key.replace('/', '-seed-')}-{mode}.npz"; np.savez_compressed(path, **saved)
        output[mode] = {**summary, "path": str(path), "sha256": sha256(path)}
    return output


def final_metrics(context: dict, arrays: dict, reward: np.ndarray, logits: dict) -> tuple[dict, dict]:
    reference = mixture(context, arrays, logits["reference"])
    evaluations, vectors, rankings = {}, {}, {}
    for name, value in logits.items():
        p = mixture(context, arrays, value)
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, reward, p, reference=reference)
    evaluations["hard-repeat"], vectors["hard-repeat"], rankings["hard-repeat"] = evaluate(context, arrays, reward, hard_repeat(arrays, reference), reference=reference)
    for name, ranking in control_rankings(arrays, context["catalog"], 20).items():
        evaluations[name], vectors[name], rankings[name] = evaluate(context, arrays, reward, rankings=ranking)
    paired = {name: intervals(context, value, vectors["reference"], arrays["user_ids"]) for name, value in vectors.items() if name != "reference"}
    margin = context["config"]["evaluation"]["ndcg_noninferiority_margin"]*evaluations["reference"]["ranking"]["metrics"]["ndcg@10"]
    primary = {"reward_improvement": paired["selected"]["reward"]["ci95"][0] > 0,
               "ndcg_noninferiority": paired["selected"]["ndcg@10"]["ci95"][0] >= -margin,
               "absolute_ndcg_margin": margin}
    primary["both_passed"] = primary["reward_improvement"] and primary["ndcg_noninferiority"]
    return {"policies": evaluations, "intervals_vs_reference": paired, "primary_confirmation": primary}, rankings


def reward_regression(context: dict, arrays: dict, reward: np.ndarray) -> tuple[dict, dict]:
    observed = attach_observed(context, arrays)
    features = history_features({**context, "train": context["priors"], "config": {"history": {"events": 64, "user_smoothing": 5, "candidate_smoothing": 5, "item_smoothing": 20}}}, observed, False)["features"]
    direct_model = CatBoostRanker(); direct_model.load_model(context["ranker_run"]["checkpoint"])
    direct = calibrated(direct_model.predict(features, thread_count=context["config"]["execution"]["threads"]), features, context["calibration"])
    cached = reward[observed["query_indices"], observed["candidate_tokens"]-3]
    difference = float(np.max(np.abs(direct-cached)))
    if difference > context["config"]["evaluation"]["forward_tolerance"]: raise ValueError("Final catalog reward differs from observed forward.")
    predictions = {"calibrated": cached, "history_user": features[:, 0], "history_candidate": features[:, 9], "item_prior": features[:, 8]}
    return {"metrics": {name: regression_metrics(observed["rewards"], p) for name, p in predictions.items()}, "maximum_direct_difference": difference}, observed


def final_evaluation(context: dict, selected: dict, runs: dict) -> dict:
    root = context["output"] / "final"
    if (root / "result.json").exists(): return read_json(root / "result.json")
    if root.exists() and any(root.iterdir()): raise FileExistsError("Preserve incomplete ranked-policy final evaluation.")
    root.mkdir()
    opening = {"opened_at_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": sha256(context["output"] / "selection.json")}
    write_json(root / "opening.json", opening); emit(stage="ranked_final_opening", selected=selected["selected_variant"])
    arrays = load_npz(context["final_data_dir"] / "final.npz")
    reward = catalog_reward(context, arrays)
    policy = new_policy(context)
    logits = {"reference": policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])}
    policy.load_state_dict(torch.load(runs[f"{selected['selected_variant']}/42"]["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    logits["selected"] = policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    evaluation, rankings = final_metrics(context, arrays, reward, logits)
    regression, observed = reward_regression(context, arrays, reward)
    for name, values in (("cache", {"reward": reward, **logits}), ("rankings", rankings), ("observed", observed)):
        np.savez_compressed(root / f"{name}.npz", **values)
    result = {**evaluation, "reward_model": regression, "opening": opening, "queries": len(arrays["lengths"]),
              "users": len(np.unique(arrays["user_ids"])), "observed_pairs": len(observed["rewards"]),
              "artifacts": {p.name: sha256(p) for p in (root / "opening.json", root / "cache.npz", root / "rankings.npz", root / "observed.npz")},
              "final_used_for_selection": False, "causal_user_effect_measured": False}
    write_json(root / "result.json", result); return result


def run_series(context: dict, caches: dict) -> dict:
    root = context["output"]
    if (root / "result.json").exists(): return read_json(root / "result.json")
    started = time.perf_counter(); controls, _ = validation_controls(context, caches)
    rollouts = {"reference/42": save_rollouts(context, caches, "reference/42", None),
                "hard-repeat/42": save_rollouts(context, caches, "hard-repeat/42", None, True)}
    runs = {}
    for variant in context["config"]["variants"]:
        key = f"{variant['name']}/42"; runs[key] = train(context, caches, variant, 42)
        rollouts[key] = save_rollouts(context, caches, key, runs[key])
    selected = choose(context, runs, controls["reference"], rollouts)
    if selected["screening_challenger"]:
        variant = next(v for v in context["config"]["variants"] if v["name"] == selected["screening_challenger"])
        for seed in context["config"]["training"]["confirmation_seeds"]:
            key = f"{variant['name']}/{seed}"; runs[key] = train(context, caches, variant, seed)
            rollouts[key] = save_rollouts(context, caches, key, runs[key])
    selected = choose(context, runs, controls["reference"], rollouts)
    selected["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "selection.json", selected); emit(stage="ranked_policy_selection", **selected)
    final = final_evaluation(context, selected, runs) if selected["accepted_on_development"] else None
    verify(context)
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"],
              "cache": read_json(root / "cache/manifest.json"), "controls": controls, "runs": runs,
              "rollouts": rollouts, "selection": selected, "final": final, "series_seconds": time.perf_counter()-started,
              "final_reserve_opened": final is not None, "rl_trained": True,
              "reward_history_during_rollout": "Fixed original observed response history, no synthetic response labels."}
    write_json(root / "result.json", result)
    emit(stage="ranked_policy_complete", runs=len(runs), selected=selected["selected_variant"],
         final_reserve_opened=final is not None, primary=final["primary_confirmation"] if final else None)
    return result


def audit(context: dict, caches: dict) -> dict:
    root = context["output"]; result = read_json(root / "result.json")
    tolerance = context["config"]["evaluation"]["forward_tolerance"]; maximum = 0.0
    reference_policy = new_policy(context)
    for split in ("train", "validation"):
        r = catalog_reward(context, context[split]); logits = policy_logits(reference_policy, context[split], context["device"], context["config"]["execution"]["inference_batch_size"])
        if np.max(np.abs(r-caches[split]["reward"])) > tolerance or np.max(np.abs(logits-caches[split]["reference_logits"])) > tolerance: raise ValueError("Ranked-policy reference/reward cache mismatch.")
    controls, _ = validation_controls(context, caches)
    if controls != result["controls"]: raise ValueError("Ranked-policy validation controls mismatch.")
    for key, record in result["runs"].items():
        path = Path(record["checkpoint"]).parent
        for name, expected in record["artifacts"].items():
            if sha256(path / name) != expected: raise ValueError("Ranked-policy training artifact changed.")
        if record["epochs"] != context["config"]["training"]["epochs"] or read_json(path / "training-history.json")["epochs"] != record["history"]: raise ValueError("Ranked-policy epoch budget mismatch.")
        policy = new_policy(context); policy.load_state_dict(torch.load(record["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        logits = policy_logits(policy, context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
        saved = load_npz(path / "validation.npz"); difference = float(np.max(np.abs(logits-saved["logits"]))); maximum = max(maximum, difference)
        if difference > tolerance: raise ValueError("Ranked-policy trained forward mismatch.")
        reference = mixture(context, context["validation"], caches["validation"]["reference_logits"])
        value, _, ranking = evaluate(context, context["validation"], caches["validation"]["reward"], mixture(context, context["validation"], logits), reference=reference)
        if value != record["validation"] or not np.array_equal(ranking, saved["rankings"]): raise ValueError("Ranked-policy metrics mismatch.")
        emit(stage="ranked_policy_audit_forward", run=key, maximum_difference=difference)
    for key, modes in result["rollouts"].items():
        record = result["runs"].get(key)
        for mode, entry in modes.items():
            saved, summary = rollout(context, caches["validation"], record, mode, key == "hard-repeat/42")
            path = Path(entry["path"]); actual = load_npz(path)
            if sha256(path) != entry["sha256"] or any(not np.array_equal(actual[k], v) for k, v in saved.items()) or summary != {k: v for k, v in entry.items() if k not in ("path", "sha256")}:
                raise ValueError("Ranked-policy rollout replay mismatch.")
    selected = choose(context, result["runs"], controls["reference"], result["rollouts"])
    if selected != {k: v for k, v in result["selection"].items() if k != "recorded_at_utc"}: raise ValueError("Ranked-policy selection mismatch.")
    if result["final"]:
        location = root / "final"; report = read_json(location / "result.json")
        if report != result["final"] or report["opening"]["opened_at_utc"] <= result["selection"]["recorded_at_utc"] or report["opening"]["selection_sha256"] != sha256(root / "selection.json"):
            raise ValueError("Ranked-policy final provenance mismatch.")
        for name, expected in report["artifacts"].items():
            if sha256(location / name) != expected: raise ValueError("Ranked-policy final artifact changed.")
        arrays = load_npz(context["final_data_dir"] / "final.npz"); saved = load_npz(location / "cache.npz")
        reward = catalog_reward(context, arrays)
        logits = {"reference": policy_logits(reference_policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])}
        policy = new_policy(context); policy.load_state_dict(torch.load(result["runs"][f"{selected['selected_variant']}/42"]["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        logits["selected"] = policy_logits(policy, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        if np.max(np.abs(reward-saved["reward"])) > tolerance or any(np.max(np.abs(v-saved[k])) > tolerance for k, v in logits.items()): raise ValueError("Ranked-policy final forward mismatch.")
        evaluation, rankings = final_metrics(context, arrays, reward, logits); regression, observed = reward_regression(context, arrays, reward)
        stored_observed = load_npz(location / "observed.npz"); stored_rankings = load_npz(location / "rankings.npz")
        if any(not np.array_equal(v, stored_observed[k]) for k, v in observed.items()) or any(not np.array_equal(v, stored_rankings[k]) for k, v in rankings.items()): raise ValueError("Final populations/rankings mismatch.")
        if any(evaluation[k] != report[k] for k in evaluation) or regression != report["reward_model"]: raise ValueError("Final metrics/intervals mismatch.")
    verify(context)
    return {"stage": "ranked_policy_audit_complete", "runs_checked": len(result["runs"]), "maximum_policy_forward_difference": maximum,
            "catalog_rewards_and_controls_recalculated": True, "rollouts_replayed": sum(len(m) for m in result["rollouts"].values()),
            "selection_recalculated": True, "final_recalculated": result["final"] is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a recency-mixture policy with a confirmed calibrated candidate ranker.")
    parser.add_argument("--config", type=Path, default=Path("configs/ranked-policy.json")); parser.add_argument("--protocol", type=Path, default=Path("experiments/ranked-policy-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ranked-policy-v1")); parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(); context = load_context(args.config, args.protocol, args.output_dir, args.device, args.resume or args.audit)
    caches = prepare_cache(context)
    if args.prepare_only: emit(stage="ranked_policy_prepared", direct_reward_difference=read_json(context["output"] / "cache/manifest.json")["maximum_direct_reward_difference"], final_reserve_opened=False)
    elif args.audit: emit(**audit(context, caches))
    else: run_series(context, caches)


if __name__ == "__main__": main()
