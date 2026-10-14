from __future__ import annotations

import math

import numpy as np

from recsys_rl.evaluation import evaluate_rankings, targets_for
from recsys_rl.reward_signal import cluster_interval


def log_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def reward_values(raw: np.ndarray, personal: np.ndarray, calibration: dict, duration_scale: np.ndarray) -> dict:
    calibrated = np.clip(calibration["slope"] * raw.astype(np.float64) + calibration["intercept"], 0, 1)
    hybrid = personal[:, None] + calibration["personal_mix_alpha"] * (calibrated - personal[:, None])
    return {"raw_completion": raw, "hybrid_completion": hybrid, "capped_time": hybrid * duration_scale[None, :]}


def query_metrics(arrays: dict, rankings: np.ndarray) -> dict:
    recall, ndcg = [], []
    for index, row in enumerate(rankings[:, :10]):
        target = targets_for(arrays, index)
        hits = np.isin(row, target)
        recall.append(float(hits.sum() / len(target)))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(target))))
        ndcg.append(sum(float(hit) / math.log2(rank + 2) for rank, hit in enumerate(hits)) / ideal)
    return {"recall@10": np.asarray(recall), "ndcg@10": np.asarray(ndcg)}


def evaluate_policy(arrays: dict, rankings: np.ndarray, catalog: dict, rewards: dict, counts: np.ndarray,
                    known_users: np.ndarray, reference_logp: np.ndarray, logits: np.ndarray | None = None,
                    distribution: np.ndarray | None = None, low_support: int = 10, observed: dict | None = None) -> tuple[dict, dict]:
    quality = evaluate_rankings(arrays, rankings, catalog, [5, 10, 20])
    vectors = query_metrics(arrays, rankings)
    row_indices, selected = np.arange(len(rankings))[:, None], rankings[:, :10] - 3
    list_rewards = {}
    for name, values in rewards.items():
        vectors[name] = values[row_indices, selected].mean(axis=1)
        list_rewards[name] = float(vectors[name].mean())
    sparse = counts <= low_support
    seen_users = np.isin(arrays["user_ids"], known_users)
    seen_targets = np.asarray([np.all(counts[targets_for(arrays, index) - 3] > 0) for index in range(len(rankings))])
    slices = {"known_reward_user": seen_users, "unknown_reward_user": ~seen_users,
              "all_targets_seen_by_reward": seen_targets, "some_target_unseen_by_reward": ~seen_targets}
    group_values = {name: {"queries": int(mask.sum()), **{key: float(value[mask].mean()) if mask.any() else None
                                                        for key, value in vectors.items()}}
                    for name, mask in slices.items()}
    result = {"ranking": quality, "top10_proxy_reward": list_rewards, "slices": group_values,
              "low_support_threshold_pairs": low_support, "low_support_top10_fraction": float(sparse[selected].mean()),
              "zero_support_top10_fraction": float((counts[selected] == 0).mean()), "policy_distribution": None}
    if logits is not None or distribution is not None:
        if logits is not None:
            logp = log_probabilities(logits)
            probabilities = np.exp(logp)
        else:
            probabilities = distribution.astype(np.float64)
            logp = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid policy distribution.")
        expected = {name: float(np.mean(np.sum(probabilities * value, axis=1))) for name, value in rewards.items()}
        result["policy_distribution"] = {
            "expected_proxy_reward": expected,
            "mean_kl_to_reference": float(np.mean(np.sum(probabilities * (logp - reference_logp), axis=1))),
            "mean_entropy": float(np.mean(-np.sum(probabilities * logp, axis=1))),
            "low_support_probability_mass": float(probabilities[:, sparse].sum(axis=1).mean()),
            "zero_support_probability_mass": float(probabilities[:, counts == 0].sum(axis=1).mean()),
            "low_support_reward_fraction": {name: float(np.sum(probabilities[:, sparse] * value[:, sparse]) /
                                                         np.sum(probabilities * value)) if np.sum(probabilities * value) > 0 else None
                                             for name, value in rewards.items()}}
    if observed is not None:
        query = observed["query_indices"]
        matched = np.any(rankings[query, :10] == observed["candidate_tokens"][:, None], axis=1)
        labels = observed["rewards"].astype(np.float64)
        result["observed_matches_only"] = {
            "observed_pairs": len(labels), "matched_pairs": int(matched.sum()),
            "matched_pair_fraction": float(matched.mean()), "queries_with_match": len(np.unique(query[matched])),
            "mean_observed_completion_on_matches": float(labels[matched].mean()) if matched.any() else None,
            "fraction_of_observed_completion_mass_captured": float(labels[matched].sum() / labels.sum()) if labels.sum() > 0 else None,
            "scope": "Conditional on observed matching listens; not an unbiased response or policy-value estimate."}
    return result, vectors


def paired_policy_intervals(vectors: dict, baseline: dict, users: np.ndarray, repeats: int, seed: int) -> dict:
    return {name: cluster_interval(vectors[name] - baseline[name], users, repeats, seed)
            for name in ("ndcg@10", "recall@10", "hybrid_completion")}


def summarize_rollout(arrays: dict, saved: dict, catalog: dict, counts: np.ndarray, support_threshold: int) -> dict:
    actions = saved["actions"]
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan
                            for item in catalog["items"]])
    current = arrays["input_ids"].copy()
    lengths = arrays["lengths"].copy()
    steps = []
    for step in range(actions.shape[1]):
        repeats, unknown = [], []
        for index, length in enumerate(lengths):
            history = current[index, 1:int(length)]
            repeats.append(actions[index, step] in history)
            unknown.append(float(np.mean(history == 2)))
        frequencies = np.bincount(actions[:, step] - 3, minlength=len(durations))
        selected_durations = durations[actions[:, step] - 3]
        steps.append({"step": step + 1, "raw_reward": float(saved["raw_reward"][:, step].mean()),
                      "hybrid_reward": float(saved["hybrid_reward"][:, step].mean()), "time_reward": float(saved["time_reward"][:, step].mean()),
                      "history_repeat_fraction": float(np.mean(repeats)), "history_unk_fraction": float(np.mean(unknown)),
                      "consecutive_generated_repeat_fraction": float(np.mean(actions[:, step] == actions[:, step - 1])) if step else None,
                      "catalog_coverage": float(np.count_nonzero(frequencies) / len(durations)),
                      "top10_items_share": float(np.sort(frequencies)[-10:].sum() / len(actions)),
                      "mean_duration_seconds": float(np.nanmean(selected_durations)) if np.isfinite(selected_durations).any() else None,
                      "low_support_action_fraction": float(np.mean(counts[actions[:, step] - 3] <= support_threshold))})
        current, lengths = append_synthetic_items(current, lengths, actions[:, step])
    return {"contexts": len(actions), "steps": steps, "mean_unique_items_per_sequence": float(np.mean([len(np.unique(row)) for row in actions])),
            "synthetic_feedback": True, "observed_response_evaluated": False}


def append_synthetic_items(inputs: np.ndarray, lengths: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output, new_lengths = np.zeros_like(inputs), np.empty_like(lengths)
    capacity = inputs.shape[1] - 1
    output[:, 0] = 1
    for index, length in enumerate(lengths):
        history = np.r_[inputs[index, 1:int(length)], actions[index]][-capacity:]
        output[index, 1:len(history) + 1] = history
        new_lengths[index] = len(history) + 1
    return output, new_lengths
