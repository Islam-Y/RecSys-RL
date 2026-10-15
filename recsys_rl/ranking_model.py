from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostRanker, Pool
from torch import nn
from torch.nn import functional

from recsys_rl.data import sha256, write_json
from recsys_rl.model import seed_everything
from recsys_rl.recency_data import read_json
from recsys_rl.reward_signal import candidate_pairs, cluster_interval


def emit(**record) -> None:
    print(json.dumps(record), flush=True)


def comparisons(context: dict, arrays: dict, capped: bool = False, strong: bool = True) -> dict:
    settings = context["config"]
    left, right = candidate_pairs(arrays, settings["evaluation"]["label_tolerance"])
    if strong:
        gap = np.abs(arrays["rewards"][left].astype(np.float64)-arrays["rewards"][right])
        keep = gap >= settings["training"]["minimum_label_gap"]-settings["evaluation"]["label_tolerance"]
        left, right = left[keep], right[keep]
    if capped and left.size:
        rng = np.random.default_rng(settings["training"]["pair_seed"])
        query = arrays["query_indices"][left]
        boundaries = np.r_[0, np.flatnonzero(np.diff(query))+1, len(query)]
        selected = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            n = min(int(stop-start), settings["training"]["maximum_pairs_per_query"])
            selected.extend(np.sort(rng.choice(np.arange(start, stop), n, replace=False)).tolist())
        selected = np.asarray(selected, dtype=np.int64); left, right = left[selected], right[selected]
    queries, inverse = np.unique(arrays["query_indices"][left], return_inverse=True)
    users = np.empty(len(queries), dtype=np.uint32)
    if queries.size: users[inverse] = arrays["user_ids"][left]
    counts = np.bincount(inverse)
    weights = (1/counts[inverse]).astype(np.float64) if left.size else np.empty(0)
    winner = np.where(arrays["rewards"][left] > arrays["rewards"][right], left, right)
    loser = np.where(arrays["rewards"][left] > arrays["rewards"][right], right, left)
    return {"left": left, "right": right, "inverse": inverse, "queries": queries, "users": users,
            "winner": winner, "loser": loser, "weights": weights}


def accuracies(context: dict, rows: dict, index: dict, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = index["left"], index["right"]
    delta = scores[left].astype(np.float64)-scores[right]
    outcome = rows["rewards"][left].astype(np.float64)-rows["rewards"][right]
    pair = np.where(np.abs(delta) <= context["config"]["evaluation"]["prediction_tolerance"], .5,
                    (np.sign(delta) == np.sign(outcome)).astype(np.float64))
    macro = np.bincount(index["inverse"], weights=pair)/np.bincount(index["inverse"]) if pair.size else np.empty(0)
    return pair, macro


def quick(context: dict, rows: dict, index: dict, scores: np.ndarray) -> float:
    macro = accuracies(context, rows, index, scores)[1]
    if not macro.size: raise ValueError("Ranking development has no eligible candidate comparisons.")
    return float(macro.mean())


def query_interval(context: dict, values: np.ndarray, users: np.ndarray) -> dict | None:
    settings = context["config"]["evaluation"]
    result = cluster_interval(values, users, settings["bootstrap_repeats"], settings["bootstrap_seed"])
    if result: result["scope"] = "Percentile user-cluster bootstrap; equal informative-query weights."
    return result


def evaluate_scores(context: dict, rows: dict, scores: dict[str, np.ndarray]) -> dict:
    result = {}
    for group, strong in (("gap_010", True), ("unequal", False)):
        index = comparisons(context, rows, strong=strong)
        result[group] = {"pairs": len(index["left"]), "queries": len(index["queries"]), "users": len(np.unique(index["users"])),
                         "methods": {}}
        vectors = {}
        for name, values in scores.items():
            pair, macro = accuracies(context, rows, index, values)
            vectors[name] = macro
            result[group]["methods"][name] = {
                "query_macro": query_interval(context, macro, index["users"]),
                "pair_weighted": cluster_interval(pair, rows["user_ids"][index["left"]], context["config"]["evaluation"]["bootstrap_repeats"], context["config"]["evaluation"]["bootstrap_seed"])}
        result[group]["paired_vs_controls"] = {name: {control: query_interval(context, values-vectors[control], index["users"])
                                                       for control in ("item_prior", "history_candidate")}
                                                for name, values in vectors.items() if name not in ("item_prior", "history_candidate", "history_user")}
    repeat = (rows["input_ids"] == rows["candidate_tokens"][:, None]).any(axis=1)
    base = comparisons(context, rows)
    slices = {"both_in_history": repeat[base["left"]] & repeat[base["right"]],
              "both_new_to_history": ~repeat[base["left"]] & ~repeat[base["right"]],
              "mixed": repeat[base["left"]] != repeat[base["right"]]}
    result["slices"] = {}
    for label, mask in slices.items():
        query, inverse = np.unique(rows["query_indices"][base["left"]][mask], return_inverse=True)
        index = {**base, "left": base["left"][mask], "right": base["right"][mask], "inverse": inverse}
        result["slices"][label] = {"pairs": int(mask.sum()), "queries": len(query),
                                   "query_macro": {name: float(accuracies(context, rows, index, values)[1].mean()) if query.size else None for name, values in scores.items()}}
    return result


class NumericRanker(nn.Module):
    def __init__(self, settings: dict):
        super().__init__()
        first, second = settings["hidden_sizes"]
        self.network = nn.Sequential(nn.Linear(16, first), nn.GELU(), nn.Dropout(settings["dropout"]),
                                     nn.Linear(first, second), nn.GELU(), nn.Linear(second, 1))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@torch.no_grad()
def neural_scores(model: NumericRanker, features: np.ndarray) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(features[start:start+8192])).numpy() for start in range(0, len(features), 8192)]).astype(np.float64)


def predict(context: dict, run: dict, features: np.ndarray) -> np.ndarray:
    path = Path(run["checkpoint"])
    if sha256(path) != run["artifacts"][path.name]: raise ValueError("Ranking checkpoint changed.")
    if run["variant"]["model"] == "boosting":
        model = CatBoostRanker(); model.load_model(str(path))
        scores = model.predict(features, thread_count=context["config"]["execution"]["threads"])
    else:
        model = NumericRanker(context["config"]["neural"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        scores = neural_scores(model, features)
    if not np.isfinite(scores).all(): raise ValueError("Non-finite rank scores.")
    return scores.astype(np.float64)


def train(context: dict, arrays: dict, features: dict, variant: dict, seed: int) -> dict:
    root = context["output"] / "runs" / variant["name"] / f"seed-{seed}"
    if (root / "summary.json").exists(): return read_json(root / "summary.json")
    if root.exists() and any(root.iterdir()): raise FileExistsError("Preserve incomplete ranking run.")
    root.mkdir(parents=True, exist_ok=True)
    settings = context["config"]; split = variant["data"]
    rows, x = arrays[split], features[split]
    pairs = comparisons(context, rows, capped=True); dev_pairs = comparisons(context, arrays["development"])
    if not pairs["left"].size: raise ValueError("No informative ranking train pairs.")
    started = time.perf_counter(); history = []; best = -float("inf")
    seed_everything(seed)
    if variant["model"] == "neural":
        model = NumericRanker(settings["neural"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["neural"]["learning_rate"], weight_decay=settings["neural"]["weight_decay"])
        tensor = torch.from_numpy(x); stale = 0
        normalized_weights = pairs["weights"]*len(pairs["left"])/len(pairs["queries"])
        for epoch in range(1, settings["neural"]["epochs"]+1):
            model.train(); order = np.random.default_rng(seed+epoch).permutation(len(pairs["left"])); total = 0.0
            for start in range(0, len(order), settings["neural"]["batch_size"]):
                selected = order[start:start+settings["neural"]["batch_size"]]
                first, second = pairs["winner"][selected], pairs["loser"][selected]
                values = model(tensor[np.r_[first, second]])
                loss = (functional.softplus(-(values[:len(first)]-values[len(first):]))*torch.as_tensor(normalized_weights[selected], dtype=torch.float32)).mean()
                if not torch.isfinite(loss): raise ValueError("Non-finite ranking loss.")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["neural"]["gradient_clip"]); optimizer.step()
                total += float(loss.detach())*len(selected)
            scores = neural_scores(model, features["development"])
            accuracy = quick(context, arrays["development"], dev_pairs, scores)
            record = {"checkpoint_step": epoch, "train_query_weighted_pair_loss": total/len(order), "development_query_macro": accuracy}
            history.append(record)
            if accuracy > best:
                best, best_step, stale = accuracy, epoch, 0
                torch.save(model.state_dict(), root / "ranker.pt"); np.save(root / "development.npy", scores, allow_pickle=False)
            else: stale += 1
            emit(stage="ranking_epoch", variant=variant["name"], seed=seed, **record)
            if stale >= settings["neural"]["patience"]: break
        checkpoint = root / "ranker.pt"
        parameters = sum(p.numel() for p in model.parameters())
    else:
        options = settings["boosting"]
        pool = Pool(x, label=rows["rewards"], group_id=rows["query_indices"],
                    pairs=np.column_stack((pairs["winner"], pairs["loser"])), pairs_weight=pairs["weights"])
        model = CatBoostRanker(iterations=options["iterations"], depth=options["depth"], learning_rate=options["learning_rate"],
                              l2_leaf_reg=options["l2_leaf_reg"], loss_function=options["loss"], random_seed=seed,
                              task_type="CPU", thread_count=options["thread_count"], verbose=False, allow_writing_files=False)
        emit(stage="ranking_boosting_started", variant=variant["name"], seed=seed, pairs=len(pairs["left"]))
        model.fit(pool)
        for number, scores in enumerate(model.staged_predict(features["development"], eval_period=options["checkpoint_period"], thread_count=options["thread_count"]), 1):
            step = min(number*options["checkpoint_period"], options["iterations"])
            accuracy = quick(context, arrays["development"], dev_pairs, scores)
            record = {"checkpoint_step": step, "development_query_macro": accuracy}; history.append(record)
            if accuracy > best:
                best, best_step = accuracy, step
                np.save(root / "development.npy", scores, allow_pickle=False)
            emit(stage="ranking_prefix", variant=variant["name"], seed=seed, **record)
        model.shrink(best_step); checkpoint = root / "ranker.cbm"; model.save_model(str(checkpoint))
        parameters = None
    write_json(root / "training-history.json", {"checkpoints": history})
    result = {"variant": variant, "seed": seed, "checkpoint": str(checkpoint), "selected_step": best_step,
              "history": history, "development_query_macro": best, "train_pairs_after_cap": len(pairs["left"]),
              "train_informative_queries": len(pairs["queries"]), "train_observed_rows": len(x), "train_users": len(np.unique(rows["user_ids"])),
              "parameters": parameters, "seconds": time.perf_counter()-started,
              "artifacts": {p.name: sha256(p) for p in (checkpoint, root / "development.npy", root / "training-history.json")}}
    write_json(root / "summary.json", result)
    return result


def fit_calibration(context: dict, scores: np.ndarray, features: np.ndarray, labels: np.ndarray) -> dict:
    center, scale = float(scores.mean()), max(float(scores.std()), 1e-6)
    z = torch.as_tensor((scores-center)/scale, dtype=torch.float64)
    user = np.clip(features[:, 0].astype(np.float64), .0001, .9999)
    h = torch.as_tensor(np.log(user/(1-user)), dtype=torch.float64); y = torch.as_tensor(labels.astype(np.float64))
    parameters = nn.Parameter(torch.tensor([np.log(np.expm1(1.0)), 1.0, 0.0], dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameters], max_iter=context["config"]["calibration"]["iterations"], line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = functional.mse_loss(torch.sigmoid(functional.softplus(parameters[0])*z+parameters[1]*h+parameters[2]), y)
        loss.backward(); return loss
    optimizer.step(closure)
    values = parameters.detach().numpy()
    return {"score_center": center, "score_scale": scale, "positive_slope": float(np.logaddexp(0, values[0])),
            "history_logit_weight": float(values[1]), "intercept": float(values[2]), "pairs": len(labels),
            "iterations_limit": context["config"]["calibration"]["iterations"]}


def calibrated(scores: np.ndarray, features: np.ndarray, settings: dict) -> np.ndarray:
    user = np.clip(features[:, 0].astype(np.float64), .0001, .9999)
    z = settings["positive_slope"]*(scores-settings["score_center"])/settings["score_scale"]
    z += settings["history_logit_weight"]*np.log(user/(1-user))+settings["intercept"]
    return np.exp(-np.logaddexp(0, -z))
