from __future__ import annotations

import math

import numpy as np


def targets_for(arrays: dict, index: int) -> np.ndarray:
    return arrays["target_tokens"][arrays["target_offsets"][index]:arrays["target_offsets"][index + 1]]


def control_rankings(arrays: dict, catalog: dict, top_n: int) -> dict[str, np.ndarray]:
    popular = np.asarray([item["token"] for item in catalog["items"]], dtype=np.int32)
    recent = []
    for row, length in zip(arrays["input_ids"], arrays["lengths"]):
        seen = set()
        ranking = []
        for token in [*row[1:int(length)][::-1], *popular]:
            token = int(token)
            if token >= 3 and token not in seen:
                seen.add(token)
                ranking.append(token)
                if len(ranking) == top_n:
                    break
        recent.append(ranking)
    return {"train_popularity": np.broadcast_to(popular[:top_n], (len(recent), top_n)), "recent_items": np.asarray(recent, dtype=np.int32)}


def evaluate_rankings(arrays: dict, rankings: np.ndarray, catalog: dict, top_k: list[int]) -> dict:
    if rankings.shape != (len(arrays["lengths"]), max(top_k)):
        raise ValueError("Ranking shape differs from the evaluation population.")
    n_items = len(catalog["items"])
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan for item in catalog["items"]])
    known_durations = durations[np.isfinite(durations)]
    boundaries = np.quantile(known_durations, [1 / 3, 2 / 3]) if len(known_durations) else [0, 0]
    values = {f"{metric}@{k}": [] for k in top_k for metric in ("recall", "ndcg")}
    duration_slices = {group: {"queries": 0, "recall@10": [], "ndcg@10": []} for group in ("short", "medium", "long", "mixed", "unknown")}
    repeats, emitted = 0, []
    for index, ranking in enumerate(rankings):
        if len(np.unique(ranking)) != len(ranking) or np.any(ranking < 3) or np.any(ranking >= n_items + 3):
            raise ValueError("Ranking contains duplicates or invalid candidate tokens.")
        target = targets_for(arrays, index)
        hits = np.isin(ranking, target)
        query_values = {}
        for k in top_k:
            recall = float(hits[:k].sum() / len(target))
            idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(target))))
            ndcg = sum(float(hit) / math.log2(rank + 2) for rank, hit in enumerate(hits[:k])) / idcg
            query_values[f"recall@{k}"] = recall
            query_values[f"ndcg@{k}"] = ndcg
            values[f"recall@{k}"].append(recall)
            values[f"ndcg@{k}"].append(ndcg)
        target_durations = durations[target - 3]
        if not np.isfinite(target_durations).all():
            group = "unknown"
        else:
            categories = np.searchsorted(boundaries, target_durations, side="right")
            group = ("short", "medium", "long")[int(categories[0])] if np.all(categories == categories[0]) else "mixed"
        duration_slices[group]["queries"] += 1
        for metric in ("recall@10", "ndcg@10"):
            duration_slices[group][metric].append(query_values[metric])
        selected = ranking[:10]
        history = arrays["input_ids"][index, 1:int(arrays["lengths"][index])]
        repeats += int(np.isin(selected, history).sum())
        emitted.extend(selected.tolist())
    emitted = np.asarray(emitted, dtype=np.int32)
    frequencies = np.bincount(emitted - 3, minlength=n_items)
    recommended_durations = durations[emitted - 3]
    return {"queries": len(rankings), "users": int(len(np.unique(arrays["user_ids"]))), "metrics": {key: float(np.mean(value)) for key, value in values.items()},
        "behavior_at_10": {"catalog_coverage": float(np.count_nonzero(frequencies) / n_items),
            "top_10_items_share": float(np.sort(frequencies)[-10:].sum() / len(emitted)),
            "context_repeat_fraction": repeats / len(emitted),
            "mean_duration_seconds": float(np.nanmean(recommended_durations)) if np.isfinite(recommended_durations).any() else None,
            "duration_known_fraction": float(np.isfinite(recommended_durations).mean())},
        "target_duration_tertiles_seconds": [float(value) for value in boundaries],
        "target_duration_slices": {key: {"queries": value["queries"], **{metric: float(np.mean(value[metric])) if value[metric] else None for metric in ("recall@10", "ndcg@10")}} for key, value in duration_slices.items()}}
