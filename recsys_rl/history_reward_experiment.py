from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from recsys_rl.data import load_partition, sha256, write_json
from recsys_rl.evaluation import control_rankings, evaluate_rankings
from recsys_rl.history_reward_data import attach_observed, build_cohort, connection, history_features, load_context, load_npz, prepare, verify
from recsys_rl.history_reward_model import HistoryCompletion, batch, predict
from recsys_rl.model import seed_everything
from recsys_rl.policy_evaluation import query_metrics
from recsys_rl.recency_data import read_json
from recsys_rl.recency_experiment import load_gate
from recsys_rl.recency_model import RecencyGate, mixture_predictions
from recsys_rl.reward import predict_reward
from recsys_rl.reward_evaluation import control_predictions, regression_metrics
from recsys_rl.reward_model import CompletionReward
from recsys_rl.reward_signal import candidate_pairs, cluster_interval, transformed


def emit(**record) -> None:
    print(json.dumps(record), flush=True)


def frozen_features(context: dict, split: str, arrays: dict) -> dict:
    root = context["output"] / "features"; root.mkdir(exist_ok=True)
    path = root / f"{split}.npz"
    if path.exists():
        return load_npz(path)
    value = history_features(context, arrays, split == "train")
    np.savez_compressed(path, **value)
    emit(stage="history_features", split=split, pairs=len(value["features"]))
    return value


def old_predictions(context: dict, arrays: dict) -> dict:
    checkpoint_path = next(Path(p) for p in context["plan"]["frozen"]["inputs"] if p.endswith("outputs/reward-v1/reward.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    users = checkpoint["state_dict"]["known_users"].numpy().astype(np.uint32)
    model = CompletionReward(context["catalog"], 64, users, context["old_reward_config"]["model"]).to(context["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    raw = predict_reward(model, arrays, users, context["device"], context["config"]["execution"]["inference_batch_size"])
    controls = control_predictions(context["train"], arrays)
    controls["raw"] = raw
    controls.update(transformed(raw, controls["user_mean"], context["calibration"]))
    return controls


def simple_controls(context: dict, arrays: dict, features: np.ndarray) -> dict:
    values = old_predictions(context, arrays)
    priors = np.clip(features[:, [0, 8]].astype(np.float64), .0001, .9999)
    logit = np.log(priors/(1-priors)).mean(axis=1)
    values.update(history_user=features[:, 0], item_smooth=features[:, 8],
                  history_candidate=features[:, 9], history_blend=1/(1+np.exp(-logit)))
    return values


def pair_statistics(context: dict, arrays: dict, predictions: dict) -> dict:
    settings = context["config"]["evaluation"]
    left, right = candidate_pairs(arrays, settings["label_tolerance"])
    differences = arrays["rewards"][left].astype(np.float64)-arrays["rewards"][right]
    result = {}
    for group, mask in {"unequal": np.ones(len(left), dtype=bool), "gap_010": np.abs(differences) >= settings["strong_label_gap"]-settings["label_tolerance"]}.items():
        first, second, outcome = left[mask], right[mask], differences[mask]
        queries, index = np.unique(arrays["query_indices"][first], return_inverse=True)
        query_users = np.empty(len(queries), dtype=np.uint32)
        if queries.size:
            query_users[index] = arrays["user_ids"][first]
        scores = {}
        for name, values in predictions.items():
            delta = values[first].astype(np.float64)-values[second]
            accuracy = np.where(np.abs(delta) <= settings["prediction_tolerance"], .5, (np.sign(delta) == np.sign(outcome)).astype(np.float64))
            macro = np.bincount(index, weights=accuracy)/np.bincount(index) if accuracy.size else np.empty(0)
            scores[name] = {"pair_weighted": cluster_interval(accuracy, arrays["user_ids"][first], settings["bootstrap_repeats"], settings["bootstrap_seed"]),
                            "query_macro": cluster_interval(macro, query_users, settings["bootstrap_repeats"], settings["bootstrap_seed"])}
            if scores[name]["query_macro"]:
                scores[name]["query_macro"]["scope"] = "User-cluster bootstrap, equal eligible-query weights."
        result[group] = {"pairs": len(first), "queries": len(queries), "users": len(np.unique(query_users)), "accuracy": scores}
    return result


def evaluate(context: dict, arrays: dict, predictions: dict) -> dict:
    settings = context["config"]["evaluation"]
    base = predictions["history_user"].astype(np.float64)
    y = arrays["rewards"].astype(np.float64)
    return {"regression": {name: regression_metrics(y, value) for name, value in predictions.items()},
            "mse_intervals_vs_history_user": {name: cluster_interval((value.astype(np.float64)-y)**2-(base-y)**2,
                                               arrays["user_ids"], settings["bootstrap_repeats"], settings["bootstrap_seed"])
                                             for name, value in predictions.items() if name.startswith("history-")},
            "pairwise": pair_statistics(context, arrays, predictions),
            "pairs": len(y), "users": len(np.unique(arrays["user_ids"]))}


def train_model(context: dict, variant: dict, features: dict) -> dict:
    root = context["output"] / "runs" / variant["name"]
    if (root / "summary.json").exists():
        return read_json(root / "summary.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete history reward run.")
    root.mkdir(parents=True, exist_ok=True)
    settings = context["config"]["training"]
    seed_everything(settings["seed"])
    model = HistoryCompletion(len(context["catalog"]["items"]), variant["kind"]).to(context["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    train, selection = context["train"], context["selection"]
    left, right = candidate_pairs(train, context["config"]["evaluation"]["label_tolerance"])
    if variant["pair_weight"] > 0 and not left.size:
        raise ValueError("Pairwise recipe has no observed training pairs.")
    started = time.perf_counter(); history = []; best = float("inf"); stale = 0
    for epoch in range(1, settings["epochs"]+1):
        model.train()
        rng = np.random.default_rng(settings["seed"]+epoch)
        order = rng.permutation(len(train["rewards"]))
        totals, pair_totals = 0.0, 0.0
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start+settings["batch_size"]]
            logits = model(*batch(train, features["train"]["features"], indices, context["device"]))
            labels = torch.as_tensor(train["rewards"][indices], device=context["device"])
            mse = functional.mse_loss(torch.sigmoid(logits), labels)
            pair_loss = torch.zeros((), device=context["device"])
            if variant["pair_weight"] > 0:
                sample = rng.integers(0, len(left), settings["pair_batch_size"])
                first, second = left[sample], right[sample]
                combined = np.r_[first, second]
                scores = model(*batch(train, features["train"]["features"], combined, context["device"]))
                delta = scores[:len(first)]-scores[len(first):]
                sign = torch.as_tensor(np.sign(train["rewards"][first]-train["rewards"][second]), device=context["device"])
                pair_loss = functional.softplus(-sign*delta).mean()
            loss = mse+variant["pair_weight"]*pair_loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite history reward loss.")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]); optimizer.step()
            totals += float(mse.detach().cpu())*len(indices); pair_totals += float(pair_loss.detach().cpu())*len(indices)
        predictions = predict(model, selection, features["selection"]["features"], context["device"], settings["batch_size"])
        metric = regression_metrics(selection["rewards"], predictions)
        record = {"epoch": epoch, "train_mse": totals/len(order), "train_pair_loss": pair_totals/len(order), "selection_mse": metric["mse"]}
        history.append(record)
        if metric["mse"] < best:
            best, best_epoch, stale = metric["mse"], epoch, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, root / "reward.pt")
            np.save(root / "selection.npy", predictions, allow_pickle=False)
        else:
            stale += 1
        write_json(root / "training-history.json", {"epochs": history})
        emit(stage="history_reward_epoch", variant=variant["name"], **record)
        if stale >= settings["patience"]:
            break
    result = {"variant": variant, "seed": settings["seed"], "parameters": sum(p.numel() for p in model.parameters()),
              "selected_epoch": best_epoch, "epochs_completed": len(history), "history": history,
              "seconds": time.perf_counter()-started, "checkpoint": str(root / "reward.pt"),
              "training_observed_comparison_pairs": len(left),
              "artifacts": {p.name: sha256(p) for p in (root / "reward.pt", root / "selection.npy", root / "training-history.json")}}
    write_json(root / "summary.json", result)
    return result


def loaded_model(context: dict, record: dict) -> HistoryCompletion:
    path = Path(record["checkpoint"])
    if sha256(path) != record["artifacts"]["reward.pt"]:
        raise ValueError("History reward weights changed.")
    model = HistoryCompletion(len(context["catalog"]["items"]), record["variant"]["kind"]).to(context["device"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def fixed_mixtures(context: dict) -> dict:
    root = context["output"] / "fixed-mixtures"; root.mkdir(exist_ok=True)
    learned = context["reference"]["runs"]["global-16/42"]
    gates = {"learned": load_gate(learned)}
    for fraction in context["config"]["fixed_mixtures"]["controls"]:
        gate = RecencyGate("global")
        with torch.no_grad(): gate.bias.copy_(torch.logit(torch.tensor(fraction, dtype=torch.float32)))
        gates[f"fixed-{fraction:.6f}"] = gate.eval()
    result = {}
    for split, arrays, path in (("validation", load_partition(context["data_dir"], "validation"), context["recency_root"] / "cache/validation.npy"),
                                ("opened-recency-holdout", load_npz(context["previous"] / "queries.npz"), context["recency_root"] / "holdout/baseline.npy")):
        logits = np.load(path, mmap_mode="r", allow_pickle=False)
        rankings = {"baseline": np.argsort(-logits, axis=1, kind="stable")[:, :20]+3, **control_rankings(arrays, context["catalog"], 20)}
        for name, gate in gates.items():
            rankings[name] = mixture_predictions(gate, arrays, logits, context["config"]["fixed_mixtures"]["tau"])[0]
        metrics = {name: evaluate_rankings(arrays, value, context["catalog"], [5, 10, 20]) for name, value in rankings.items()}
        vectors = {name: query_metrics(arrays, value) for name, value in rankings.items()}
        settings = context["config"]["evaluation"]
        comparisons = {name: {metric: cluster_interval(vectors[name][metric]-vectors["learned"][metric], arrays["user_ids"], settings["bootstrap_repeats"], settings["bootstrap_seed"])
                              for metric in ("ndcg@10", "recall@10")} for name in gates if name != "learned"}
        np.savez_compressed(root / f"{split}.npz", **rankings)
        result[split] = {"methods": metrics, "intervals_vs_learned": comparisons,
                         "sha256": sha256(root / f"{split}.npz"), "independent_confirmation": False}
    return result


def choose(context: dict, seen: dict, unseen: dict, runs: dict) -> dict:
    selection = context["config"]["selection"]
    reasons, candidates = {}, []
    for name in runs:
        metric = unseen["regression"][name]["mse"]
        pairwise = unseen["pairwise"]["gap_010"]
        macro = pairwise["accuracy"][name]["query_macro"]
        item = pairwise["accuracy"]["item_smooth"]["query_macro"]
        checks = {"mse_below_original_raw_hybrid_and_history_user": all(metric < unseen["regression"][control]["mse"] for control in ("raw", "hybrid", "global_mean", "history_user")),
                  "enough_pair_queries": pairwise["queries"] >= selection["minimum_pair_queries"],
                  "query_macro_at_least_055": macro is not None and macro["estimate"] >= selection["minimum_query_macro_accuracy"],
                  "query_macro_not_below_item_control": macro is not None and item is not None and macro["estimate"] >= item["estimate"],
                  "seen_mse_preserved": seen["regression"][name]["mse"] <= selection["maximum_seen_mse_ratio"]*min(seen["regression"][c]["mse"] for c in ("raw", "hybrid", "history_user"))}
        reasons[name] = checks
        if all(checks.values()): candidates.append(name)
    selected = min(candidates, key=lambda n: unseen["regression"][n]["mse"]) if candidates else None
    return {"selected_reward": selected, "rl_allowed": selected is not None, "eligibility": reasons,
            "checkpoint": {"path": runs[selected]["checkpoint"], "sha256": runs[selected]["artifacts"]["reward.pt"]} if selected else None,
            "final_reserve_opened": False}


def run(context: dict, preparation: dict) -> dict:
    root = context["output"]
    if (root / "result.json").exists(): return read_json(root / "result.json")
    started = time.perf_counter()
    mixtures = fixed_mixtures(context)
    features = {s: frozen_features(context, s, context[s]) for s in ("train", "selection")}
    runs = {v["name"]: train_model(context, v, features) for v in context["config"]["variants"]}
    write_json(root / "checkpoints-frozen.json", {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "runs": runs})
    selection_predictions = simple_controls(context, context["selection"], features["selection"]["features"])
    for name, record in runs.items(): selection_predictions[name] = np.load(Path(record["checkpoint"]).parent / "selection.npy", allow_pickle=False)
    seen = evaluate(context, context["selection"], selection_predictions)
    opened = datetime.now(timezone.utc).isoformat()
    write_json(root / "development-opening.json", {"opened_at_utc": opened, "checkpoints_sha256": sha256(root / "checkpoints-frozen.json")})
    emit(stage="history_reward_development_opening")
    queries = load_npz(context["new_data_dir"] / "development.npz")
    observed = attach_observed(context, queries)
    np.savez_compressed(root / "development-observed.npz", **observed)
    development_features = frozen_features(context, "development", observed)
    predictions = simple_controls(context, observed, development_features["features"])
    for name, record in runs.items():
        predictions[name] = predict(loaded_model(context, record), observed, development_features["features"], context["device"], context["config"]["execution"]["inference_batch_size"])
    np.savez_compressed(root / "selection-predictions.npz", **selection_predictions)
    np.savez_compressed(root / "development-predictions.npz", **predictions)
    unseen = evaluate(context, observed, predictions)
    selected = choose(context, seen, unseen, runs)
    selected["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "selection.json", selected)
    verify(context)
    files = [root / name for name in ("checkpoints-frozen.json", "development-opening.json", "development-observed.npz", "selection-predictions.npz", "development-predictions.npz", "selection.json")]
    files.extend(sorted((root / "features").glob("*.npz")))
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "preparation": preparation,
              "fixed_mixtures": mixtures, "runs": runs, "seen_selection": seen, "unseen_development": unseen,
              "selection": selected, "seconds": time.perf_counter()-started,
              "artifacts": {str(p.relative_to(root)): sha256(p) for p in files}, "final_reserve_opened": False,
              "rl_trained": False, "limitations": ["New development users used for recipe selection, not independent test.",
                  "Past observed response is additional input; architecture-only attribution is not possible.",
                  "Train item priors use leave-one-out encoding, not out-of-fold or sequential fitting.",
                  "Reward regression and observational candidate comparisons do not establish causal policy value."]}
    write_json(root / "result.json", result)
    return result


def audit(context: dict, preparation: dict) -> dict:
    root = context["output"]; result = read_json(root / "result.json")
    for name, expected in result["artifacts"].items():
        if sha256(root / name) != expected: raise ValueError("History run artifact changed.")
    if result["provenance"] != context["plan"] or result["preparation"] != preparation:
        raise ValueError("History result provenance changed.")
    cohorts = load_npz(context["new_data_dir"] / "development-users.npz")
    db = connection(context)
    try:
        for split, excluded, settings in (("development", cohorts["excluded_users"], context["config"]["reward_development"]),
                                          ("final", np.union1d(cohorts["excluded_users"], cohorts["users"]), context["config"]["final_reserve"])):
            arrays, users = build_cohort(context, db, settings, excluded)
            for filename, values in ((f"{split}.npz", arrays), (f"{split}-users.npz", users)):
                saved = load_npz(context["new_data_dir"] / filename)
                if saved.keys() != values.keys() or any(not np.array_equal(saved[k], v) for k, v in values.items()):
                    raise ValueError("Raw cohort reconstruction mismatch.")
    finally: db.close()
    observed = attach_observed(context, load_npz(context["new_data_dir"] / "development.npz"))
    saved = load_npz(root / "development-observed.npz")
    if any(not np.array_equal(saved[k], v) for k, v in observed.items()): raise ValueError("Development labels mismatch.")
    measured = {}
    maximum = 0.0
    for split, arrays in (("train", context["train"]), ("selection", context["selection"]), ("development", observed)):
        features = history_features(context, arrays, split == "train")
        saved_features = load_npz(root / "features" / f"{split}.npz")
        if any(not np.array_equal(saved_features[k], v) for k, v in features.items()): raise ValueError("Past-response features do not reconstruct.")
        if split == "train": continue
        predictions = simple_controls(context, arrays, features["features"])
        for name, record in result["runs"].items():
            run_root = Path(record["checkpoint"]).parent
            for filename, expected in record["artifacts"].items():
                if sha256(run_root / filename) != expected: raise ValueError("History reward checkpoint evidence changed.")
            history = read_json(run_root / "training-history.json")["epochs"]
            if min(history, key=lambda r: r["selection_mse"])["epoch"] != record["selected_epoch"] or history != record["history"]:
                raise ValueError("Reward checkpoint selection mismatch.")
            predictions[name] = predict(loaded_model(context, record), arrays, features["features"], context["device"], context["config"]["execution"]["inference_batch_size"])
        saved = load_npz(root / f"{split}-predictions.npz")
        for name, value in predictions.items():
            difference = float(np.max(np.abs(value-saved[name])))
            maximum = max(maximum, difference)
            if difference > context["config"]["evaluation"]["forward_tolerance"]: raise ValueError("History reward forward mismatch.")
        measured[split] = evaluate(context, arrays, predictions)
        emit(stage="history_reward_audit", split=split, maximum_difference=maximum)
    if measured["selection"] != result["seen_selection"] or measured["development"] != result["unseen_development"]:
        raise ValueError("History reward metrics/intervals mismatch.")
    selection = choose(context, measured["selection"], measured["development"], result["runs"])
    if selection != {k: v for k, v in result["selection"].items() if k != "recorded_at_utc"}: raise ValueError("Reward recipe selection mismatch.")
    frozen = read_json(root / "checkpoints-frozen.json"); opening = read_json(root / "development-opening.json")
    if opening["opened_at_utc"] <= frozen["recorded_at_utc"] or opening["checkpoints_sha256"] != sha256(root / "checkpoints-frozen.json"):
        raise ValueError("Development opened before checkpoint freeze.")
    if fixed_mixtures(context) != result["fixed_mixtures"]: raise ValueError("Fixed mixture results mismatch.")
    verify(context)
    return {"stage": "history_reward_audit_complete", "reward_runs": len(result["runs"]), "maximum_forward_difference": maximum,
            "raw_cohorts_and_response_histories_reconstructed": True, "all_metrics_and_selection_recalculated": True,
            "final_reserve_opened": False, "rl_allowed": selection["rl_allowed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fixed mixtures and history-based reward models before new RL.")
    parser.add_argument("--config", type=Path, default=Path("configs/reward-history.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/reward-history-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reward-history-v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config, args.protocol, args.output_dir, args.device, args.resume or args.audit)
    preparation = prepare(context)
    if args.prepare_only: emit(stage="history_reward_prepared", development=preparation["development"], final_reserve=preparation["final_reserve"], response_labels_read=False)
    elif args.audit: emit(**audit(context, preparation))
    else:
        result = run(context, preparation)
        emit(stage="history_reward_complete", runs=len(result["runs"]), **result["selection"])


if __name__ == "__main__": main()
