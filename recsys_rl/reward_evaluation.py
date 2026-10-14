from __future__ import annotations

import numpy as np

from recsys_rl.reward_model import user_tokens


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    if not labels.size:
        return {"pairs": 0, "mse": None, "rmse": None, "mae": None, "r2": None, "pearson": None,
                "bias": None, "mean_prediction": None, "mean_observed": None}
    labels, predictions = labels.astype(np.float64), predictions.astype(np.float64)
    errors = predictions - labels
    mse = float(np.mean(errors ** 2))
    variance = float(np.mean((labels - labels.mean()) ** 2))
    pearson = None
    if np.ptp(labels) > 0 and np.ptp(predictions) > 0:
        centered_labels, centered_predictions = labels - labels.mean(), predictions - predictions.mean()
        pearson = float(np.dot(centered_labels, centered_predictions) /
                        np.sqrt(np.dot(centered_labels, centered_labels) * np.dot(centered_predictions, centered_predictions)))
    return {"pairs": len(labels), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(errors))),
            "r2": 1 - mse / variance if variance > 0 else None, "pearson": pearson,
            "bias": float(errors.mean()), "mean_prediction": float(predictions.mean()), "mean_observed": float(labels.mean())}


def control_predictions(train: dict, arrays: dict) -> dict:
    labels = train["rewards"].astype(np.float64)
    global_mean = float(labels.mean())
    users = np.unique(train["user_ids"])
    train_users = user_tokens(train["user_ids"], users)

    def group_means(indices: np.ndarray) -> np.ndarray:
        counts = np.bincount(indices)
        totals = np.bincount(indices, weights=labels)
        return np.divide(totals, counts, out=np.full(len(counts), global_mean), where=counts > 0)

    item_means, user_means = group_means(train["candidate_tokens"]), group_means(train_users)
    candidates = arrays["candidate_tokens"]
    item_predictions = np.full(len(candidates), global_mean)
    available = candidates < len(item_means)
    item_predictions[available] = item_means[candidates[available]]
    return {"global_mean": np.full(len(candidates), global_mean), "item_mean": item_predictions,
            "user_mean": user_means[user_tokens(arrays["user_ids"], users)]}


def evaluate_reward(arrays: dict, predictions: np.ndarray, catalog: dict, known_users: np.ndarray,
                    known_items: np.ndarray, bins: int) -> dict:
    if predictions.shape != arrays["rewards"].shape or not np.all(np.isfinite(predictions)):
        raise ValueError("Invalid reward prediction shape or values.")
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Completion reward prediction is outside [0, 1].")
    labels = arrays["rewards"]
    all_durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan
                                for item in catalog["items"]], dtype=np.float64)
    known_durations = all_durations[np.isfinite(all_durations) & (all_durations > 0)]
    cuts = np.quantile(known_durations, [1 / 3, 2 / 3]) if known_durations.size else np.asarray([0.0, 0.0])
    durations = all_durations[arrays["candidate_tokens"] - 3]
    duration_known = np.isfinite(durations) & (durations > 0)
    seen_users = user_tokens(arrays["user_ids"], known_users) > 0
    seen_items = np.isin(arrays["candidate_tokens"], known_items)
    repeats = (arrays["input_ids"] == arrays["candidate_tokens"][:, None]).any(axis=1)
    groups = {"duration": {"short": duration_known & (durations <= cuts[0]),
                           "medium": duration_known & (durations > cuts[0]) & (durations <= cuts[1]),
                           "long": duration_known & (durations > cuts[1]), "unknown": ~duration_known},
              "organic": {"recommendation_driven": arrays["organic"] == 0, "organic": arrays["organic"] == 1,
                          "mixed_or_unknown": arrays["organic"] == 2},
              "user": {"seen": seen_users, "unseen": ~seen_users},
              "item": {"seen": seen_items, "unseen": ~seen_items},
              "context_repeat": {"repeat": repeats, "new_to_context": ~repeats}}
    calibration = []
    assignments = np.minimum((predictions * bins).astype(np.int64), bins - 1)
    for index in range(bins):
        selected = assignments == index
        metrics = regression_metrics(labels[selected], predictions[selected])
        calibration.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "pairs": metrics["pairs"], "mean_prediction": metrics["mean_prediction"],
                            "mean_observed": metrics["mean_observed"]})
    _, inverse = np.unique(arrays["user_ids"], return_inverse=True)
    errors = predictions.astype(np.float64) - labels.astype(np.float64)
    user_mse = np.bincount(inverse, weights=errors ** 2) / np.bincount(inverse)
    return {"metrics": regression_metrics(labels, predictions), "users": len(user_mse),
            "user_macro_mse": float(user_mse.mean()), "duration_tertiles_seconds": cuts.tolist(),
            "slices": {name: {group: regression_metrics(labels[mask], predictions[mask]) for group, mask in values.items()}
                       for name, values in groups.items()}, "calibration": calibration}
