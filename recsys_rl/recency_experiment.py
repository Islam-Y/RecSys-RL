from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings, evaluate_rankings, targets_for
from recsys_rl.model import seed_everything
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.policy_evaluation import append_synthetic_items, query_metrics
from recsys_rl.recency_data import build_holdout, load_context, prepare_holdout, read_json, verify_inputs
from recsys_rl.recency_model import RecencyGate, mixture_predictions, mixture_probabilities, training_inputs
from recsys_rl.reward_signal import cluster_interval


def emit(**record) -> None:
    print(json.dumps(record), flush=True)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def prepare_cache(context: dict) -> dict:
    root = context["output"] / "cache"
    root.mkdir(exist_ok=True)
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Recency cache changed.")
    else:
        if any(root.iterdir()):
            raise FileExistsError("Incomplete recency cache must be preserved.")
        started = time.perf_counter()
        policy = new_policy(context)
        for split in ("train", "validation"):
            arrays = context[split]
            destination = np.lib.format.open_memmap(root / f"{split}.npy", mode="w+", dtype=np.float32,
                                                   shape=(len(arrays["lengths"]), len(context["catalog"]["items"])))
            block = 2048
            for start in range(0, len(arrays["lengths"]), block):
                rows = {key: arrays[key][start:start+block] for key in ("input_ids", "lengths")}
                destination[start:start+block] = policy_logits(policy, rows, context["device"], context["config"]["execution"]["inference_batch_size"])
                emit(stage="recency_cache", split=split, queries=min(start+block, len(arrays["lengths"])))
            destination.flush()
            del destination
        manifest = {"artifacts": {f"{s}.npy": sha256(root / f"{s}.npy") for s in ("train", "validation")},
                    "seconds": time.perf_counter()-started, "train_queries": len(context["train"]["lengths"])}
        write_json(root / "manifest.json", manifest)
    return {split: np.load(root / f"{split}.npy", mmap_mode="r", allow_pickle=False) for split in ("train", "validation")}


def ranking_summary(context: dict, arrays: dict, rankings: np.ndarray) -> dict:
    return evaluate_rankings(arrays, rankings, context["catalog"], context["config"]["evaluation"]["top_k"])


def quick_quality(arrays: dict, rankings: np.ndarray) -> dict:
    vectors = query_metrics(arrays, rankings)
    return {name: float(value.mean()) for name, value in vectors.items()}


def train_variant(context: dict, cache: dict, variant: dict, seed: int, prepared: dict) -> dict:
    root = context["output"] / "runs" / variant["name"] / f"seed-{seed}"
    if (root / "summary.json").exists():
        return read_json(root / "summary.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve an incomplete gate run before retrying.")
    root.mkdir(parents=True, exist_ok=True)
    settings = context["config"]["training"]
    seed_everything(seed)
    gate = RecencyGate(variant["gate"])
    optimizer = torch.optim.AdamW(gate.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    tensors = {key: torch.from_numpy(value) for key, value in prepared.items()}
    history, best, stale, best_epoch = [], -float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(1, settings["epochs"]+1):
        gate.train()
        order = np.random.default_rng(seed+epoch-1).permutation(len(prepared["features"]))
        total_loss = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start+settings["batch_size"]]
            g = gate(tensors["features"][indices])*tensors["available"][indices]
            probabilities = (1-g[:, None])*tensors["base"][indices] + g[:, None]*tensors["copy"][indices]
            per_target = -torch.log(probabilities.clamp_min(1e-38))*tensors["mask"][indices]
            loss = (per_target.sum(dim=1)/tensors["mask"][indices].sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite mixture CE.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), settings["gradient_clip"])
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        rankings, gates, ce = mixture_predictions(gate, context["validation"], cache["validation"], variant["tau"])
        quality = quick_quality(context["validation"], rankings)
        history.append({"epoch": epoch, "train_ce": total_loss/len(order), "validation_ce": float(ce.mean()),
                        **quality, "mean_gate": float(gates.mean())})
        if quality["ndcg@10"] > best:
            best, best_epoch, stale = quality["ndcg@10"], epoch, 0
            torch.save(gate.state_dict(), root / "gate.pt")
            np.savez_compressed(root / "validation.npz", rankings=rankings, gates=gates, ce=ce)
        else:
            stale += 1
        write_json(root / "training-history.json", {"epochs": history})
        emit(stage="recency_epoch", variant=variant["name"], seed=seed, **history[-1])
        if stale >= settings["patience"]:
            break
    saved = load_npz(root / "validation.npz")
    result = {"variant": variant, "seed": seed, "parameters": sum(p.numel() for p in gate.parameters()),
              "checkpoint": str(root / "gate.pt"), "checkpoint_sha256": sha256(root / "gate.pt"),
              "validation_sha256": sha256(root / "validation.npz"), "history_sha256": sha256(root / "training-history.json"),
              "selected_epoch": best_epoch, "epochs_completed": len(history), "training_history": history,
              "seconds": time.perf_counter()-started, "validation": ranking_summary(context, context["validation"], saved["rankings"]),
              "gate_mean": float(saved["gates"].mean()), "gate_quantiles": np.quantile(saved["gates"], [0, .1, .5, .9, 1]).tolist(),
              "validation_ce": float(saved["ce"].mean())}
    write_json(root / "summary.json", result)
    return result


def controls(context: dict, arrays: dict, logits: np.ndarray) -> tuple[dict, dict]:
    rankings = {"baseline": (np.argsort(-logits, axis=1, kind="stable")[:, :20]+3).astype(np.int32),
                **control_rankings(arrays, context["catalog"], 20)}
    return {name: ranking_summary(context, arrays, row) for name, row in rankings.items()}, rankings


def eligible(context: dict, run: dict, comparison: dict) -> bool:
    criteria = context["config"]["selection"]
    metric = run["validation"]["metrics"]
    return all(metric["ndcg@10"] >= criteria["minimum_relative_ndcg"]*comparison[name]["metrics"]["ndcg@10"] and
               metric["recall@10"] >= criteria["minimum_relative_recall"]*comparison[name]["metrics"]["recall@10"]
               for name in ("baseline", "recent_items"))


def decision(context: dict, runs: dict, comparison: dict) -> dict:
    initial = [runs[f"{v['name']}/42"] for v in context["config"]["variants"]]
    allowed = [r for r in initial if eligible(context, r, comparison)]
    challenger = max(allowed, key=lambda r: r["validation"]["metrics"]["ndcg@10"]) if allowed else None
    name = challenger["variant"]["name"] if challenger else None
    seeds = [context["config"]["training"]["seed"], *context["config"]["training"]["confirmation_seeds"]]
    confirmations = [{"seed": s, "eligible": eligible(context, runs[f"{name}/{s}"], comparison)}
                     for s in seeds if name and f"{name}/{s}" in runs]
    accepted = name is not None and len(confirmations) == len(seeds) and all(r["eligible"] for r in confirmations)
    return {"screening_challenger": name, "confirmation": confirmations,
            "selected_variant": name if accepted else "baseline", "accepted_on_development": accepted,
            "checkpoint": {"path": challenger["checkpoint"], "sha256": challenger["checkpoint_sha256"]} if accepted else None,
            "holdout_used_for_selection": False}


def load_gate(record: dict) -> RecencyGate:
    if sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
        raise ValueError("Gate checkpoint changed.")
    gate = RecencyGate(record["variant"]["gate"])
    gate.load_state_dict(torch.load(record["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    return gate.eval()


def intervals(context: dict, arrays: dict, candidate: np.ndarray, reference: np.ndarray) -> dict:
    values, base = query_metrics(arrays, candidate), query_metrics(arrays, reference)
    settings = context["config"]["evaluation"]
    result = {k: cluster_interval(values[k]-base[k], arrays["user_ids"], settings["bootstrap_repeats"], settings["bootstrap_seed"]) for k in values}
    for item in result.values():
        item["scope"] = "Percentile user-cluster bootstrap; equal query weights in the observed cohort."
    return result


def evaluation_slices(arrays: dict, rankings: np.ndarray) -> dict:
    metrics = query_metrics(arrays, rankings)
    repeat = np.asarray([np.all(np.isin(targets_for(arrays, i), arrays["input_ids"][i, 1:int(arrays["lengths"][i])])) for i in range(len(rankings))])
    return {name: {"queries": int(mask.sum()), **{k: float(v[mask].mean()) if mask.any() else None for k, v in metrics.items()}}
            for name, mask in {"all_targets_in_history": repeat, "some_target_new_to_history": ~repeat}.items()}


def evaluate_holdout(context: dict, runs: dict, selected: dict, preparation: dict) -> dict | None:
    if not selected["accepted_on_development"]:
        return None
    root = context["output"] / "holdout"
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete holdout evaluation before retrying.")
    root.mkdir()
    opened = datetime.now(timezone.utc).isoformat()
    write_json(root / "opening.json", {"opened_at_utc": opened, "selection_sha256": sha256(context["output"] / "selection.json")})
    emit(stage="recency_holdout_opening", selected=selected["selected_variant"])
    arrays = load_npz(context["holdout_dir"] / "queries.npz")
    logits = policy_logits(new_policy(context), arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
    np.save(root / "baseline.npy", logits, allow_pickle=False)
    results, rankings = controls(context, arrays, logits)
    chosen = runs[f"{selected['selected_variant']}/42"]
    evaluated = [chosen]
    if chosen["variant"]["gate"] == "context":
        evaluated.append(runs[f"global-{int(chosen['variant']['tau'])}/42"])
    for run in evaluated:
        name = run["variant"]["name"]
        ranking, gates, ce = mixture_predictions(load_gate(run), arrays, logits, run["variant"]["tau"])
        rankings[name] = ranking
        results[name] = {**ranking_summary(context, arrays, ranking), "gate_mean": float(gates.mean()),
                         "gate_quantiles": np.quantile(gates, [0, .1, .5, .9, 1]).tolist(), "ce": float(ce.mean())}
    name = selected["selected_variant"]
    comparisons = {control: intervals(context, arrays, rankings[name], rankings[control]) for control in ("baseline", "recent_items")}
    primary = {"ndcg_superiority_vs_baseline": comparisons["baseline"]["ndcg@10"]["ci95"][0] > 0,
               "ndcg_superiority_vs_recent": comparisons["recent_items"]["ndcg@10"]["ci95"][0] > 0}
    primary["both_passed"] = all(primary.values())
    result = {"opened_at_utc": opened, "selection_sha256": sha256(context["output"] / "selection.json"),
              "preparation": preparation, "policies": results, "intervals_for_selected": comparisons, "primary_confirmation": primary,
              "slices": {k: evaluation_slices(arrays, v) for k, v in rankings.items()}, "response_labels_read": False,
              "limitations": "Previously unused users, same source/time/catalog; no causal user-benefit evaluation."}
    np.savez_compressed(root / "rankings.npz", **rankings)
    result["artifacts"] = {p.name: sha256(p) for p in (root / "baseline.npy", root / "rankings.npz", root / "opening.json")}
    write_json(root / "result.json", result)
    return result


def rollouts(context: dict, run: dict) -> dict:
    settings = context["config"]["rollout"]
    arrays = context["validation"]
    rng = np.random.default_rng(settings["seed"])
    indices = np.sort(rng.choice(len(arrays["lengths"]), settings["contexts"], replace=False))
    uniforms = rng.random((len(indices), settings["steps"]))
    policy, gate = new_policy(context), load_gate(run)
    output = {}
    for name, mode in (("baseline", "greedy"), ("baseline", "sampled"), (run["variant"]["name"], "greedy"),
                       (run["variant"]["name"], "sampled"), ("recent_items", "greedy")):
        current = {k: arrays[k][indices].copy() for k in ("input_ids", "lengths")}
        actions = np.empty((len(indices), settings["steps"]), dtype=np.int32)
        for step in range(settings["steps"]):
            if name == "recent_items":
                action = control_rankings(current, context["catalog"], 20)["recent_items"][:, 0]
            else:
                logits = policy_logits(policy, current, context["device"], context["config"]["execution"]["inference_batch_size"])
                if name == "baseline":
                    shifted = logits.astype(np.float64)-logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
                else:
                    probabilities = mixture_probabilities(gate, current, logits, run["variant"]["tau"]).astype(np.float64)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                if mode == "greedy":
                    action = np.argmax(probabilities, axis=1)+3
                else:
                    action = np.minimum((np.cumsum(probabilities, axis=1) < uniforms[:, step, None]).sum(axis=1), probabilities.shape[1]-1)+3
            actions[:, step] = action
            current["input_ids"], current["lengths"] = append_synthetic_items(current["input_ids"], current["lengths"], action)
        key = f"{name}-{mode}"
        output[key] = {"actions": actions, "summary": {"contexts": len(indices), "synthetic_feedback": True,
                       "mean_unique_items": float(np.mean([len(np.unique(row)) for row in actions])),
                       "consecutive_repeat_by_step": [None, *[float(np.mean(actions[:, s] == actions[:, s-1])) for s in range(1, settings["steps"])]]}}
    return output


def run_series(context: dict, cache: dict, preparation: dict) -> dict:
    result_path = context["output"] / "result.json"
    if result_path.exists():
        return read_json(result_path)
    started = time.perf_counter()
    comparison, _ = controls(context, context["validation"], cache["validation"])
    runs, prepared_by_tau = {}, {}
    for variant in context["config"]["variants"]:
        tau = variant["tau"]
        if tau not in prepared_by_tau:
            prepared_by_tau[tau] = training_inputs(context["train"], cache["train"], tau)
        record = train_variant(context, cache, variant, 42, prepared_by_tau[tau])
        runs[f"{variant['name']}/42"] = record
    selected = decision(context, runs, comparison)
    if selected["screening_challenger"]:
        variant = next(v for v in context["config"]["variants"] if v["name"] == selected["screening_challenger"])
        for seed in context["config"]["training"]["confirmation_seeds"]:
            runs[f"{variant['name']}/{seed}"] = train_variant(context, cache, variant, seed, prepared_by_tau[variant["tau"]])
    selected = decision(context, runs, comparison)
    selected["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(context["output"] / "selection.json", selected)
    emit(stage="recency_selection", **selected)
    holdout = evaluate_holdout(context, runs, selected, preparation)
    rollout_summaries = {}
    if selected["accepted_on_development"]:
        root = context["output"] / "rollouts"
        root.mkdir(exist_ok=True)
        for name, value in rollouts(context, runs[f"{selected['selected_variant']}/42"]).items():
            np.savez_compressed(root / f"{name}.npz", actions=value["actions"])
            rollout_summaries[name] = {**value["summary"], "sha256": sha256(root / f"{name}.npz")}
    verify_inputs(context)
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "preparation": preparation,
              "cache": read_json(context["output"] / "cache" / "manifest.json"), "validation_controls": comparison,
              "runs": runs, "selection": selected, "holdout": holdout, "rollouts": rollout_summaries,
              "series_seconds": time.perf_counter()-started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                              "backbone_device": str(context["device"]), "gate_device": "cpu"},
              "historical_test_reevaluated": False, "rl_retrained": False, "reward_retrained": False}
    write_json(result_path, result)
    return result


def audit(context: dict, cache: dict, preparation: dict) -> dict:
    result = read_json(context["output"] / "result.json")
    tolerance = context["config"]["evaluation"]["forward_tolerance"]
    policy = new_policy(context)
    differences = {}
    for split in ("train", "validation"):
        maximum = 0.0
        for start in range(0, len(context[split]["lengths"]), 2048):
            rows = {k: context[split][k][start:start+2048] for k in ("input_ids", "lengths")}
            actual = policy_logits(policy, rows, context["device"], context["config"]["execution"]["inference_batch_size"])
            maximum = max(maximum, float(np.max(np.abs(actual-cache[split][start:start+2048]))))
        if maximum > tolerance:
            raise ValueError("Frozen GPT-2 cache forward mismatch.")
        differences[split] = maximum
        emit(stage="recency_audit_forward", split=split, maximum_difference=maximum)
    comparison, _ = controls(context, context["validation"], cache["validation"])
    if comparison != result["validation_controls"]:
        raise ValueError("Validation controls changed.")
    for run in result["runs"].values():
        gate = load_gate(run)
        path = Path(run["checkpoint"]).parent
        for name, expected in (("validation.npz", run["validation_sha256"]), ("training-history.json", run["history_sha256"])):
            if sha256(path / name) != expected:
                raise ValueError("Gate run evidence changed.")
        rankings, gates, ce = mixture_predictions(gate, context["validation"], cache["validation"], run["variant"]["tau"])
        saved = load_npz(path / "validation.npz")
        if not np.array_equal(rankings, saved["rankings"]) or not np.allclose(gates, saved["gates"], atol=tolerance, rtol=0) or not np.allclose(ce, saved["ce"], atol=tolerance, rtol=0):
            raise ValueError("Gate validation forward mismatch.")
        if ranking_summary(context, context["validation"], rankings) != run["validation"]:
            raise ValueError("Gate validation metrics differ.")
        history = read_json(path / "training-history.json")["epochs"]
        best_epoch = max(history, key=lambda r: r["ndcg@10"])["epoch"]
        if best_epoch != run["selected_epoch"]:
            raise ValueError("Best epoch selection mismatch.")
    rebuilt = decision(context, result["runs"], comparison)
    saved_selection = read_json(context["output"] / "selection.json")
    if rebuilt != {k: v for k, v in saved_selection.items() if k != "recorded_at_utc"} or saved_selection != result["selection"]:
        raise ValueError("Gate selection mismatch.")
    raw_arrays, cohort, description = build_holdout(context)
    for file, expected in (("queries.npz", raw_arrays), ("users.npz", cohort)):
        saved = load_npz(context["holdout_dir"] / file)
        if saved.keys() != expected.keys() or any(not np.array_equal(saved[k], v) for k, v in expected.items()):
            raise ValueError("Raw holdout reconstruction mismatch.")
    if any(preparation[k] != v for k, v in description.items()):
        raise ValueError("Holdout description mismatch.")
    if result["holdout"]:
        root = context["output"] / "holdout"
        holdout = read_json(root / "result.json")
        if holdout != result["holdout"]:
            raise ValueError("Holdout report mismatch.")
        if holdout["opened_at_utc"] <= saved_selection["recorded_at_utc"] or holdout["selection_sha256"] != sha256(context["output"] / "selection.json"):
            raise ValueError("Holdout was not sealed until selection.")
        for name, expected in holdout["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Holdout evidence changed.")
        logits = policy_logits(policy, raw_arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        differences["holdout"] = float(np.max(np.abs(logits-np.load(root / "baseline.npy", allow_pickle=False))))
        if differences["holdout"] > tolerance:
            raise ValueError("Holdout GPT-2 mismatch.")
        summaries, rankings = controls(context, raw_arrays, logits)
        for name in holdout["policies"]:
            if name in summaries:
                continue
            run = result["runs"][f"{name}/42"]
            ranking, gates, ce = mixture_predictions(load_gate(run), raw_arrays, logits, run["variant"]["tau"])
            rankings[name] = ranking
            summaries[name] = {**ranking_summary(context, raw_arrays, ranking), "gate_mean": float(gates.mean()),
                               "gate_quantiles": np.quantile(gates, [0, .1, .5, .9, 1]).tolist(), "ce": float(ce.mean())}
        saved = load_npz(root / "rankings.npz")
        if summaries != holdout["policies"] or any(not np.array_equal(saved[k], v) for k, v in rankings.items()):
            raise ValueError("Holdout policy metrics mismatch.")
        name = result["selection"]["selected_variant"]
        comparisons = {k: intervals(context, raw_arrays, rankings[name], rankings[k]) for k in ("baseline", "recent_items")}
        if comparisons != holdout["intervals_for_selected"] or {k: evaluation_slices(raw_arrays, v) for k, v in rankings.items()} != holdout["slices"]:
            raise ValueError("Holdout intervals or slices mismatch.")
        primary = {"ndcg_superiority_vs_baseline": comparisons["baseline"]["ndcg@10"]["ci95"][0] > 0,
                   "ndcg_superiority_vs_recent": comparisons["recent_items"]["ndcg@10"]["ci95"][0] > 0}
        primary["both_passed"] = all(primary.values())
        if primary != holdout["primary_confirmation"]:
            raise ValueError("Primary confirmation mismatch.")
        for key, value in rollouts(context, result["runs"][f"{name}/42"]).items():
            path = context["output"] / "rollouts" / f"{key}.npz"
            expected = result["rollouts"][key]
            if sha256(path) != expected["sha256"] or not np.array_equal(load_npz(path)["actions"], value["actions"]) or value["summary"] != {k: v for k, v in expected.items() if k != "sha256"}:
                raise ValueError("Rollout replay mismatch.")
    verify_inputs(context)
    return {"stage": "recency_audit_complete", "runs_checked": len(result["runs"]), "backbone_forward_maximum_difference": differences,
            "selection_recalculated": True, "raw_holdout_reconstructed": True, "user_overlap": 0,
            "metrics_and_intervals_recalculated": True, "rollouts_replayed": len(result["rollouts"]), "historical_test_reevaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a recency mixture gate over a frozen next-item model.")
    parser.add_argument("--config", type=Path, default=Path("configs/recency.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/recency-protocol.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/recency-v1"))
    parser.add_argument("--holdout-dir", type=Path, default=Path("data/processed/recency-holdout-v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config, args.protocol, args.output_dir, args.holdout_dir, args.device, args.resume or args.audit)
    preparation = prepare_holdout(context)
    if args.prepare_only:
        emit(stage="recency_prepared", **{k: preparation[k] for k in ("selected_users", "excluded_users", "queries", "user_overlap", "response_labels_read")})
        return
    cache = prepare_cache(context)
    if args.audit:
        emit(**audit(context, cache, preparation))
    else:
        result = run_series(context, cache, preparation)
        emit(stage="recency_complete", runs=len(result["runs"]), selected=result["selection"]["selected_variant"],
             primary=result["holdout"]["primary_confirmation"] if result["holdout"] else None)


if __name__ == "__main__":
    main()
