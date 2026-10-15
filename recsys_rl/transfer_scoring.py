from __future__ import annotations

from pathlib import Path

import numpy as np
from catboost import CatBoostRanker

from recsys_rl.candidate_policy_experiment import policy_for
from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings, evaluate_rankings
from recsys_rl.history_reward_data import attach_observed, history_features, load_npz
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import append_synthetic_items
from recsys_rl.ranked_policy_data import catalog_reward, mixture
from recsys_rl.ranked_policy_training import evaluate, intervals
from recsys_rl.ranking_model import calibrated, emit
from recsys_rl.recency_data import read_json
from recsys_rl.reward_evaluation import regression_metrics
from recsys_rl.reward_signal import cluster_interval
from recsys_rl.shortlist_mass_experiment import contract, probabilities
from recsys_rl.transfer_data import query_slice


METHODS = ("reference", "selected", "matched_truncated", "historical_rl", "expanded_rl", "train_popularity", "recent_items")


def models(context: dict) -> dict:
    historical = {"kind": "rl", "checkpoint": str(context["historical_checkpoint"]), "checkpoint_sha256": sha256(context["historical_checkpoint"])}
    return {"reference": new_policy(context), "historical_rl": policy_for(context, historical), "expanded_rl": policy_for(context, context["expanded_descriptor"])}


def forwards(context: dict, arrays: dict, policies: dict) -> dict:
    cache = {"reward": catalog_reward(context, arrays)}
    for name, model in policies.items():
        cache[f"{name}_logits"] = policy_logits(model, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    return cache


def observed_predictions(context: dict, arrays: dict, cache: dict) -> tuple[dict, dict, float]:
    observed = attach_observed(context, arrays)
    if not observed["rewards"].size:
        return observed, {k: np.empty(0, dtype=np.float32) for k in ("labels", "calibrated", "history_user", "history_candidate", "item_prior")}, 0.0
    feature_context = {**context, "train": context["priors"], "config": {"history": {"events": 64, "user_smoothing": 5, "candidate_smoothing": 5, "item_smoothing": 20}}}
    features = history_features(feature_context, observed, False)["features"]
    ranker = CatBoostRanker()
    ranker.load_model(context["ranker_run"]["checkpoint"])
    direct = calibrated(ranker.predict(features, thread_count=context["config"]["execution"]["threads"]), features, context["calibration"])
    cached = cache["reward"][observed["query_indices"], observed["candidate_tokens"]-3]
    difference = float(np.max(np.abs(direct-cached)))
    if difference > context["config"]["evaluation"]["forward_tolerance"]:
        raise ValueError("Transfer catalog reward differs from direct observed prediction.")
    return observed, {"labels": observed["rewards"], "calibrated": cached, "history_user": features[:, 0], "history_candidate": features[:, 9], "item_prior": features[:, 8]}, difference


def chunk_values(context: dict, arrays: dict, cache: dict) -> tuple[dict, dict, dict, dict, dict]:
    reference = mixture(context, arrays, cache["reference_logits"])
    rankings, vectors, distributions, contracts = {}, {}, {}, {}
    for name in METHODS:
        if name in ("train_popularity", "recent_items"):
            continue
        if name == "reference":
            p = reference
        elif name in ("selected", "matched_truncated"):
            rule = context["rule"] if name == "selected" else {**context["rule"], "mode": "truncate"}
            p = probabilities(arrays, reference, cache["reward"], rule)
            contracts[name] = contract(context, reference, p, rule)
        else:
            p = mixture(context, arrays, cache[f"{name}_logits"])
        value, vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], p, reference=reference)
        distributions[name] = value["distribution"]
    for name, ranking in control_rankings(arrays, context["catalog"], 20).items():
        value, vectors[name], rankings[name] = evaluate(context, arrays, cache["reward"], rankings=ranking)
        distributions[name] = value["distribution"]
    identity_rule = {**context["rule"], "reward_weight": 0}
    identity = probabilities(arrays, reference, cache["reward"], identity_rule)
    contracts["identity"] = contract(context, reference, identity, identity_rule)
    observed, predictions, difference = observed_predictions(context, arrays, cache)
    metadata = {"queries": len(arrays["lengths"]), "distributions": distributions, "contracts": contracts, "maximum_direct_reward_difference": difference}
    flat_vectors = {f"{name}::{metric}": values for name, group in vectors.items() for metric, values in group.items()}
    return metadata, rankings, flat_vectors, observed, predictions


def collect_chunks(context: dict, cohort: str, arrays: dict, policies: dict, audit: bool) -> tuple[dict, dict, dict, dict, dict, list, float]:
    root = context["output"] / cohort / "chunks"
    if not audit:
        root.mkdir(parents=True, exist_ok=True)
    size = context["config"]["execution"]["query_chunk_size"]
    rng = np.random.default_rng(context["config"]["rollout"]["context_seed"])
    sample = np.sort(rng.choice(len(arrays["lengths"]), min(context["config"]["rollout"]["contexts"], len(arrays["lengths"])), replace=False))
    sample_reward = np.empty((len(sample), len(context["catalog"]["items"])), dtype=np.float32)
    rank_parts, vector_parts, prediction_parts = {}, {}, {}
    distributions = {name: None for name in METHODS}
    contracts = {}
    records = []
    maximum = 0.0
    direct_difference = 0.0
    for number, start in enumerate(range(0, len(arrays["lengths"]), size)):
        stop = min(start+size, len(arrays["lengths"]))
        part = query_slice(arrays, start, stop)
        location = root / f"{number:04d}"
        record_path = location / "summary.json"
        if record_path.exists():
            record = read_json(record_path)
            for name, expected in record["artifacts"].items():
                if sha256(location / name) != expected:
                    raise ValueError("Transfer chunk artifact changed.")
            saved_cache = load_npz(location / "forward.npz")
            if audit:
                cache = forwards(context, part, policies)
                difference = max(float(np.max(np.abs(cache[k]-saved_cache[k]))) for k in cache)
                maximum = max(maximum, difference)
                if difference > context["config"]["evaluation"]["forward_tolerance"]:
                    raise ValueError("Transfer chunk forward differs.")
                metadata, ranks, vectors, observed, predictions = chunk_values(context, part, cache)
                if metadata != record["metrics"]:
                    raise ValueError("Transfer chunk metrics or contracts differ.")
                for name, actual in (("rankings", ranks), ("vectors", vectors), ("observed", observed), ("predictions", predictions)):
                    saved = load_npz(location / f"{name}.npz")
                    if set(actual) != set(saved) or any(not np.array_equal(v, saved[k]) for k, v in actual.items()):
                        raise ValueError(f"Transfer chunk {name} differs.")
            else:
                cache = saved_cache
                metadata = record["metrics"]
                ranks, vectors, predictions = [load_npz(location / f"{name}.npz") for name in ("rankings", "vectors", "predictions")]
        else:
            if audit:
                raise FileNotFoundError("Transfer audit requires completed chunks.")
            if location.exists() and any(location.iterdir()):
                raise FileExistsError("Preserve incomplete transfer chunk.")
            location.mkdir()
            cache = forwards(context, part, policies)
            metadata, ranks, vectors, observed, predictions = chunk_values(context, part, cache)
            for name, values in (("forward", cache), ("rankings", ranks), ("vectors", vectors), ("observed", observed), ("predictions", predictions)):
                np.savez_compressed(location / f"{name}.npz", **values)
            record = {"start": start, "stop": stop, "metrics": metadata,
                      "artifacts": {p.name: sha256(p) for p in sorted(location.glob("*.npz"))}}
            write_json(record_path, record)
        if record["start"] != start or record["stop"] != stop:
            raise ValueError("Transfer chunk boundaries changed.")
        records.append({"path": str(record_path), "sha256": sha256(record_path)})
        included = (sample >= start) & (sample < stop)
        sample_reward[included] = cache["reward"][sample[included]-start]
        for name, values in ranks.items():
            rank_parts.setdefault(name, []).append(values)
        for name, values in vectors.items():
            vector_parts.setdefault(name, []).append(values)
        for name, values in predictions.items():
            prediction_parts.setdefault(name, []).append(values)
        for name, distribution in metadata["distributions"].items():
            if distribution is not None:
                if distributions[name] is None:
                    distributions[name] = {k: 0.0 for k in distribution}
                for key, value in distribution.items():
                    distributions[name][key] += value*(stop-start)/len(arrays["lengths"])
        for name, values in metadata["contracts"].items():
            existing = contracts.setdefault(name, {"maximum_outside_difference": 0.0, "maximum_inside_mass_difference": 0.0, "identical_to_reference": True})
            for key in ("maximum_outside_difference", "maximum_inside_mass_difference"):
                existing[key] = max(existing[key], values[key])
            existing["identical_to_reference"] = existing["identical_to_reference"] and values["identical_to_reference"]
        direct_difference = max(direct_difference, metadata["maximum_direct_reward_difference"])
        emit(stage="transfer_chunk", cohort=cohort, chunk=number, queries=stop, total=len(arrays["lengths"]), audit=audit)
    rankings = {k: np.concatenate(v) for k, v in rank_parts.items()}
    flat = {k: np.concatenate(v) for k, v in vector_parts.items()}
    predictions = {k: np.concatenate(v) for k, v in prediction_parts.items()}
    evaluations = {}
    for name in METHODS:
        evaluations[name] = {"ranking": evaluate_rankings(arrays, rankings[name], context["catalog"], [5, 10, 20]),
                             "top10_reward": float(flat[f"{name}::reward"].mean()), "distribution": distributions[name]}
    reward_model = {"metrics": {name: regression_metrics(predictions["labels"], p) for name, p in predictions.items() if name != "labels"}, "maximum_direct_difference": direct_difference}
    metadata = {"policies": evaluations, "contracts": contracts, "reward_model": reward_model}
    return metadata, rankings, flat, predictions, {"indices": sample, "reward": sample_reward}, records, maximum


def rollout(context: dict, arrays: dict, inputs: dict, policy, rule: dict | None, mode: str, seed: int | None) -> tuple[dict, dict]:
    indices = inputs["indices"]
    working = {k: arrays[k][indices].copy() for k in ("input_ids", "lengths")}
    settings = context["config"]["rollout"]
    uniforms = np.random.default_rng(seed).random((len(indices), settings["steps"])) if mode == "sampled" else None
    actions = np.empty((len(indices), settings["steps"]), dtype=np.int32)
    rewards = np.empty(actions.shape, dtype=np.float32)
    for step in range(settings["steps"]):
        logits = policy_logits(policy, working, context["device"], context["config"]["execution"]["inference_batch_size"])
        p = mixture(context, working, logits).astype(np.float64)
        if rule is not None:
            p = probabilities(working, p, inputs["reward"], rule)
        p /= p.sum(axis=1, keepdims=True)
        if mode == "greedy":
            action = np.argmax(p, axis=1)+3
        elif mode == "sampled":
            action = np.minimum((np.cumsum(p, axis=1) < uniforms[:, step, None]).sum(axis=1), p.shape[1]-1)+3
        else:
            raise ValueError("Unknown transfer rollout mode.")
        actions[:, step] = action
        rewards[:, step] = inputs["reward"][np.arange(len(indices)), action-3]
        working["input_ids"], working["lengths"] = append_synthetic_items(working["input_ids"], working["lengths"], action)
    summary = {"contexts": len(indices), "mean_unique": float(np.mean([len(np.unique(row)) for row in actions])),
               "repeat_step5": float(np.mean(actions[:, -1] == actions[:, -2])), "reward_by_step": rewards.mean(axis=0, dtype=np.float64).tolist(),
               "observed_response_history_frozen": True, "real_feedback_measured": False}
    return {"indices": indices, "actions": actions, "reward": rewards}, summary


def rollout_records(context: dict, cohort: str, arrays: dict, inputs: dict, policy, audit: bool) -> dict:
    root = context["output"] / cohort / "rollouts"
    if not audit:
        root.mkdir(exist_ok=True)
    results = {}
    for name, rule in (("reference", None), ("selected", context["rule"]), ("matched_truncated", {**context["rule"], "mode": "truncate"})):
        results[name] = {}
        modes = [("greedy", None), *[("sampled", seed) for seed in context["config"]["rollout"]["sampling_seeds"]]]
        for mode, seed in modes:
            key = mode if seed is None else f"{mode}-{seed}"
            path = root / f"{name}-{key}.npz"
            summary_path = path.with_suffix(".json")
            if summary_path.exists():
                record = read_json(summary_path)
                if sha256(path) != record["sha256"]:
                    raise ValueError("Transfer rollout artifact changed.")
                if audit:
                    values, summary = rollout(context, arrays, inputs, policy, rule, mode, seed)
                    saved = load_npz(path)
                    if any(not np.array_equal(v, saved[k]) for k, v in values.items()) or summary != record["metrics"]:
                        raise ValueError("Transfer rollout replay differs.")
            else:
                if audit or path.exists():
                    raise FileExistsError("Missing or incomplete transfer rollout.")
                values, summary = rollout(context, arrays, inputs, policy, rule, mode, seed)
                np.savez_compressed(path, **values)
                record = {"metrics": summary, "path": str(path), "sha256": sha256(path), "rule": rule, "sampling_seed": seed}
                write_json(summary_path, record)
            results[name][key] = record
            emit(stage="transfer_rollout", cohort=cohort, policy=name, mode=key, audit=audit)
    return results


def cohort_metrics(context: dict, arrays: dict, metadata: dict, flat_vectors: dict, records: dict) -> dict:
    vectors = {name: {metric: flat_vectors[f"{name}::{metric}"] for metric in ("ndcg@10", "recall@10", "reward")} for name in METHODS}
    paired = {name: intervals(context, value, vectors["reference"], arrays["user_ids"]) for name, value in vectors.items() if name != "reference"}
    comparisons = {name: intervals(context, vectors["selected"], vectors[name], arrays["user_ids"]) for name in ("matched_truncated", "historical_rl", "expanded_rl", "recent_items")}
    outcomes, sampled = {}, {}
    indices = None
    for name, modes in records.items():
        unique, repeat = [], []
        for seed in context["config"]["rollout"]["sampling_seeds"]:
            value = load_npz(Path(modes[f"sampled-{seed}"]["path"]))
            if indices is None:
                indices = value["indices"]
            elif not np.array_equal(indices, value["indices"]):
                raise ValueError("Transfer sampled contexts differ between methods or seeds.")
            unique.append(np.asarray([len(np.unique(row)) for row in value["actions"]], dtype=np.float64))
            repeat.append((value["actions"][:, -1] == value["actions"][:, -2]).astype(np.float64))
        outcomes[name] = {"mean_unique": np.mean(unique, axis=0), "repeat_step5": np.mean(repeat, axis=0)}
        sampled[name] = {key: float(values.mean()) for key, values in outcomes[name].items()}
    behavior = {}
    settings = context["config"]["evaluation"]
    for baseline in ("reference", "matched_truncated"):
        behavior[baseline] = {metric: cluster_interval(outcomes["selected"][metric]-outcomes[baseline][metric], arrays["user_ids"][indices], settings["bootstrap_repeats"], settings["bootstrap_seed"])
                              for metric in ("mean_unique", "repeat_step5")}
        for value in behavior[baseline].values():
            value["scope"] = "Paired user-cluster bootstrap of per-query means across three fixed sampling seeds; equal sampled-query weights."
    criteria = context["config"]["selection"]
    margin = settings["ndcg_noninferiority_margin"]*metadata["policies"]["reference"]["ranking"]["metrics"]["ndcg@10"]
    primary = {"reward_improvement": paired["selected"]["reward"]["ci95"][0] > 0,
               "ndcg_noninferiority": paired["selected"]["ndcg@10"]["ci95"][0] >= -margin,
               "ndcg_strict_improvement": paired["selected"]["ndcg@10"]["ci95"][0] > 0,
               "sampled_point_guards_passed": sampled["selected"]["mean_unique"] >= sampled["reference"]["mean_unique"]-criteria["maximum_unique_decrease"] and sampled["selected"]["repeat_step5"] <= sampled["reference"]["repeat_step5"]+criteria["maximum_sampled_repeat_increase"],
               "absolute_ndcg_margin": margin}
    primary["all_passed"] = primary["reward_improvement"] and primary["ndcg_noninferiority"] and primary["sampled_point_guards_passed"]
    return {**metadata, "intervals_vs_reference": paired, "selected_vs_controls": comparisons, "sampled_three_seed_means": sampled,
            "behavior_intervals": behavior, "confirmation": primary, "rollouts": records}
