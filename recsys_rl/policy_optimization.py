from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from recsys_rl.data import sha256, write_json
from recsys_rl.evaluation import control_rankings
from recsys_rl.model import seed_everything
from recsys_rl.policy_data import load_pilot, new_policy, policy_logits, prepare_pilot_cache, verify_pilot_inputs
from recsys_rl.policy_evaluation import evaluate_policy, log_probabilities, paired_policy_intervals, reward_values
from recsys_rl.policy_rollout import generate_rollouts
from recsys_rl.reward_data import load_reward_partition
from recsys_rl.train import batch_tensors


def effective_beta(variant: dict, context: dict) -> float:
    return variant["kl_beta_base"] * (context["affine_scale"] if variant["match_affine_scale"] else 1)


def objective_statistics(logits: np.ndarray, reward: np.ndarray, reference_logp: np.ndarray, beta: float) -> dict:
    logp = log_probabilities(logits)
    probabilities = np.exp(logp)
    expected = float(np.mean(np.sum(probabilities * reward, axis=1)))
    kl = float(np.mean(np.sum(probabilities * (logp - reference_logp), axis=1)))
    return {"expected_optimization_reward": expected, "mean_kl_to_reference": kl,
            "mean_entropy": float(np.mean(-np.sum(probabilities * logp, axis=1))), "objective": expected - beta * kl}


def validation_evaluation(context: dict, cache: dict, rewards: dict, rankings: np.ndarray,
                          logits: np.ndarray | None = None, distribution: np.ndarray | None = None) -> tuple[dict, dict]:
    return evaluate_policy(context["validation"], rankings, context["catalog"], rewards, context["counts"], context["known_users"],
                           log_probabilities(cache["reference_logits"]), logits, distribution,
                           context["config"]["evaluation"]["low_support_threshold"], context["observed_validation"])


def audit_policy_run(context: dict, cache: dict, rewards: dict, baseline_vectors: dict, output: Path, record: dict) -> None:
    if record["epochs_completed"] != context["config"]["training"]["epochs"] or record["selected_epoch"] != record["epochs_completed"]:
        raise ValueError("Policy run did not use its fixed final-epoch budget.")
    if record["test_evaluated"] or record["reference_policy_sha256"] != context["frozen"]["inputs"][str(context["policy_checkpoint"])]:
        raise ValueError("Policy reference or test flag differs.")
    for name, value in record["artifacts"].items():
        if sha256(output / name) != value["sha256"]:
            raise ValueError(f"Policy run artifact changed: {output}/{name}")
    with np.load(output / "validation.npz", allow_pickle=False) as saved:
        for name in ("user_ids", "timestamps", "target_tokens", "target_offsets"):
            if not np.array_equal(saved[name], context["validation"][name]):
                raise ValueError(f"Policy validation population differs: {name}")
        rankings = np.argsort(-saved["logits"], axis=1, kind="stable")[:, :20] + 3
        if not np.array_equal(rankings, saved["rankings"]):
            raise ValueError("Policy rankings differ from saved logits.")
        evaluation, vectors = validation_evaluation(context, cache, rewards, rankings, logits=saved["logits"])
    if evaluation != record["validation"]:
        raise ValueError("Policy evaluation does not reproduce.")
    settings = context["config"]["evaluation"]
    intervals = paired_policy_intervals(vectors, baseline_vectors, context["validation"]["user_ids"], settings["bootstrap_repeats"], settings["bootstrap_seed"])
    if intervals != record["paired_intervals_vs_baseline"]:
        raise ValueError("Policy paired intervals differ.")
    if json.loads((output / "training-history.json").read_text())["epochs"] != record["training_history"]:
        raise ValueError("Policy training history differs.")


def train_policy_variant(context: dict, caches: dict, rewards: dict, baseline_vectors: dict, variant: dict, training_seed: int) -> dict:
    name, beta = variant["name"], effective_beta(variant, context)
    root = context["output"] / "runs" / name / f"seed-{training_seed}"
    verify_pilot_inputs(context)
    summary_path = root / "summary.json"
    if summary_path.exists():
        record = json.loads(summary_path.read_text())
        if record["variant"] != variant or record["training_seed"] != training_seed or record["effective_beta"] != beta:
            raise ValueError("Saved policy variant differs from the plan.")
        audit_policy_run(context, caches["validation"], rewards["validation"], baseline_vectors, root, record)
        print(json.dumps({"stage": "rl_run_reused", "variant": name, "training_seed": training_seed}), flush=True)
        return record
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Incomplete policy run must be preserved separately before retrying: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    settings = context["config"]["training"]
    seed_everything(training_seed)
    policy = new_policy(context)
    policy.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    reference_logits = caches["train"]["reference_logits"]
    reward = rewards["train"][variant["reward"]]
    reference_logp = log_probabilities(reference_logits)
    initial = objective_statistics(reference_logits, reward, reference_logp, beta)
    history = []
    print(json.dumps({"stage": "rl_training_started", "variant": name, "training_seed": training_seed, "beta": beta,
                      "contexts": len(reward), "initial": initial}), flush=True)
    for epoch in range(settings["epochs"]):
        epoch_started = time.perf_counter()
        order = np.random.default_rng(training_seed + epoch).permutation(len(reward))
        loss_sum = 0.0
        for offset in range(0, len(order), settings["batch_size"]):
            indices = order[offset:offset + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = policy(*batch_tensors(context["train"], indices, context["device"]))
            logp = functional.log_softmax(logits, dim=1)
            probabilities = logp.exp()
            target_reward = torch.as_tensor(reward[indices].astype(np.float32), device=context["device"])
            reference = functional.log_softmax(torch.as_tensor(reference_logits[indices], device=context["device"]), dim=1)
            expected = (probabilities * target_reward).sum(dim=1)
            kl = (probabilities * (logp - reference)).sum(dim=1)
            loss = (-expected + beta * kl).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite policy optimization loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
        logits = policy_logits(policy, context["train"], context["device"], context["config"]["execution"]["inference_batch_size"])
        statistics = objective_statistics(logits, reward, reference_logp, beta)
        record = {"epoch": epoch + 1, "mean_training_loss_during_updates": loss_sum / len(order),
                  "train_after_epoch": statistics, "seconds": time.perf_counter() - epoch_started}
        history.append(record)
        write_json(root / "training-history.json", {"epochs": history})
        print(json.dumps({"stage": "rl_epoch_completed", "variant": name, "training_seed": training_seed, **record}), flush=True)
    torch.save({key: value.detach().cpu() for key, value in policy.state_dict().items()}, root / "policy.pt")
    logits = policy_logits(policy, context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
    rankings = (np.argsort(-logits, axis=1, kind="stable")[:, :20] + 3).astype(np.int32)
    np.savez_compressed(root / "validation.npz", logits=logits, rankings=rankings,
                        **{key: context["validation"][key] for key in ("user_ids", "timestamps", "target_tokens", "target_offsets")})
    evaluation, vectors = validation_evaluation(context, caches["validation"], rewards["validation"], rankings, logits=logits)
    intervals = paired_policy_intervals(vectors, baseline_vectors, context["validation"]["user_ids"],
                                        context["config"]["evaluation"]["bootstrap_repeats"], context["config"]["evaluation"]["bootstrap_seed"])
    record = {"variant": variant, "effective_beta": beta, "training_seed": training_seed, "data_seed": context["config"]["data_seed"],
              "source_commit": context["plan"]["source_commit"], "parameters": sum(parameter.numel() for parameter in policy.parameters()),
              "reference_policy_sha256": context["frozen"]["inputs"][str(context["policy_checkpoint"])], "dropout_enabled": False,
              "epochs_completed": len(history), "selected_epoch": len(history), "optimizer_steps": len(history) * int(np.ceil(len(reward) / settings["batch_size"])),
              "initial_objective": initial, "training_history": history, "validation": evaluation,
              "paired_intervals_vs_baseline": intervals, "inprocess_run_seconds": time.perf_counter() - started,
              "artifacts": {filename: {"sha256": sha256(root / filename), "bytes": (root / filename).stat().st_size}
                            for filename in ("policy.pt", "validation.npz", "training-history.json")}, "rl_trained": True, "test_evaluated": False}
    audit_policy_run(context, caches["validation"], rewards["validation"], baseline_vectors, root, record)
    record["audit"] = {"rankings_metrics_intervals_and_population_recalculated": True, "artifact_hashes_checked": True}
    write_json(summary_path, record)
    print(json.dumps({"stage": "rl_run_completed", "variant": name, "training_seed": training_seed,
                      "ndcg_at_10": evaluation["ranking"]["metrics"]["ndcg@10"],
                      "hybrid_top10_proxy_reward": evaluation["top10_proxy_reward"]["hybrid_completion"],
                      "seconds": record["inprocess_run_seconds"]}), flush=True)
    del policy, optimizer
    if context["device"].type == "mps":
        torch.mps.empty_cache()
    return record


def eligible_policy(record: dict, baseline: dict, context: dict) -> bool:
    criteria = context["config"]["selection"]
    return (record["variant"]["name"] not in criteria["exclude_variants"] and
            record["validation"]["ranking"]["metrics"]["ndcg@10"] >= criteria["minimum_relative_ndcg_at_10"] * baseline["ranking"]["metrics"]["ndcg@10"] and
            record["validation"]["top10_proxy_reward"]["hybrid_completion"] > baseline["top10_proxy_reward"]["hybrid_completion"])


def build_controls(context: dict, caches: dict, rewards: dict) -> tuple[dict, dict, dict, float]:
    reference_logits = caches["validation"]["reference_logits"]
    reference_logp = log_probabilities(reference_logits)
    baseline_rankings = np.argsort(-reference_logits, axis=1, kind="stable")[:, :20] + 3
    baseline, baseline_vectors = validation_evaluation(context, caches["validation"], rewards["validation"], baseline_rankings, logits=reference_logits)
    controls, control_arrays = {}, {"baseline": baseline_rankings}
    for name, rankings in control_rankings(context["validation"], context["catalog"], 20).items():
        controls[name], _ = validation_evaluation(context, caches["validation"], rewards["validation"], rankings)
        control_arrays[name] = rankings
    controls["baseline"] = baseline
    matched_logp = None
    matched_difference = None
    for variant in context["config"]["variants"]:
        beta = effective_beta(variant, context)
        reward = rewards["validation"][variant["reward"]]
        if beta > 0:
            logits = reference_logp + reward / beta
            rankings = np.argsort(-logits, axis=1, kind="stable")[:, :20] + 3
            evaluation, _ = validation_evaluation(context, caches["validation"], rewards["validation"], rankings, logits=logits)
            if variant["name"] == "raw-kl":
                matched_logp = log_probabilities(logits)
            if variant["name"] == "hybrid-matched-kl":
                matched_difference = float(np.max(np.abs(np.exp(log_probabilities(logits)) - np.exp(matched_logp))))
                if matched_difference > context["config"]["reward"]["matched_policy_tolerance"]:
                    raise ValueError("Matched raw/hybrid analytical policies differ.")
        else:
            rankings = np.argsort(-reward, axis=1, kind="stable")[:, :20] + 3
            point_mass = np.zeros_like(reward, dtype=np.float32)
            point_mass[np.arange(len(reward)), rankings[:, 0] - 3] = 1
            evaluation, _ = validation_evaluation(context, caches["validation"], rewards["validation"], rankings, distribution=point_mass)
        key = f"analytic_{variant['name']}"
        controls[key] = {"variant": variant, "effective_beta": beta, "evaluation": evaluation}
        control_arrays[key] = rankings
    return controls, control_arrays, baseline_vectors, matched_difference


def select_challenger(context: dict, runs: dict, baseline: dict) -> dict | None:
    seed = context["config"]["training_seed"]
    candidates = [runs[f"{variant['name']}/seed-{seed}"] for variant in context["config"]["variants"]
                  if eligible_policy(runs[f"{variant['name']}/seed-{seed}"], baseline, context)]
    return max(candidates, key=lambda row: row["validation"]["top10_proxy_reward"]["hybrid_completion"]) if candidates else None


def run_policy_pilot(data_dir: Path, reward_data_dir: Path, output_dir: Path, config_path: Path,
                     protocol_path: Path, requested_device: str, resume: bool, prepare_only: bool) -> dict:
    started = time.perf_counter()
    context = load_pilot(data_dir, reward_data_dir, output_dir, config_path, protocol_path, requested_device, resume)
    output_dir = context["output"]
    caches, cache_manifest = prepare_pilot_cache(context)
    if prepare_only:
        return {"stage": "rl_cache_ready", "cache_manifest": cache_manifest, "rl_trained": False, "test_evaluated": False}
    context["observed_validation"] = load_reward_partition(context["reward_data_dir"], "validation")
    rewards = {split: reward_values(cache["raw_reward"], cache["personal_mean"], context["calibration"], context["duration_scale"])
               for split, cache in caches.items()}
    controls, control_arrays, baseline_vectors, matched_difference = build_controls(context, caches, rewards)
    baseline = controls["baseline"]
    np.savez_compressed(output_dir / "control-rankings.npz", **control_arrays)
    runs = {}
    seed = context["config"]["training_seed"]
    for variant in context["config"]["variants"]:
        key = f"{variant['name']}/seed-{seed}"
        runs[key] = train_policy_variant(context, caches, rewards, baseline_vectors, variant, seed)
        write_json(output_dir / "progress.json", {"stage": "screening", "completed_runs": list(runs)})
    challenger = select_challenger(context, runs, baseline)
    confirmation = []
    if challenger is not None:
        for seed in context["config"]["confirmation_seeds"]:
            key = f"{challenger['variant']['name']}/seed-{seed}"
            if key not in runs:
                runs[key] = train_policy_variant(context, caches, rewards, baseline_vectors, challenger["variant"], seed)
                write_json(output_dir / "progress.json", {"stage": "confirmation", "completed_runs": list(runs)})
            confirmation.append({"training_seed": seed, "eligible": eligible_policy(runs[key], baseline, context)})
    accepted = challenger is not None and all(row["eligible"] for row in confirmation)
    selected = challenger["variant"]["name"] if accepted else "baseline"
    print(json.dumps({"stage": "rl_selection_complete", "challenger": challenger["variant"]["name"] if challenger else None,
                      "selected_variant": selected, "confirmation": confirmation}), flush=True)
    rollouts = generate_rollouts(context, runs)
    verify_pilot_inputs(context)
    for name, values in cache_manifest["artifacts"].items():
        if sha256(output_dir / "cache" / name) != values["sha256"]:
            raise ValueError(f"Pilot cache changed: {name}")
    for key, record in runs.items():
        root = output_dir / "runs" / record["variant"]["name"] / f"seed-{record['training_seed']}"
        audit_policy_run(context, caches["validation"], rewards["validation"], baseline_vectors, root, record)
    checkpoint = context["policy_checkpoint"] if not accepted else output_dir / "runs" / selected / f"seed-{context['config']['selection']['checkpoint_training_seed']}" / "policy.pt"
    result = {"experiment": context["config"]["experiment"], "provenance": context["plan"], "environment": context["environment"],
              "paths": context["paths"],
              "cache": cache_manifest, "controls": controls, "runs": runs, "screening_challenger": challenger["variant"]["name"] if challenger else None,
              "confirmation": confirmation, "selected_variant": selected,
              "selected_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
              "matched_analytic_policy_maximum_difference": matched_difference, "rollouts": rollouts,
              "control_rankings_artifact": {"path": str(output_dir / "control-rankings.npz"), "sha256": sha256(output_dir / "control-rankings.npz")},
              "current_invocation_seconds": time.perf_counter() - started,
              "sum_inprocess_training_run_seconds": sum(record["inprocess_run_seconds"] for record in runs.values()),
              "audit": {"cached_reward_forward_checked": True, "reference_evaluation_reproduced": True,
                        "all_runs_reaudited": True, "rollout_rewards_replayed": True,
                        "rollout_policy_actions_replayed": True, "frozen_inputs_unchanged": True},
              "limits": ["Model-based one-step contextual-bandit optimization, without real user interaction or multi-step return training.",
                         "Proxy reward trained on observed listens; candidate discrimination is weak and off-distribution outcomes are unknown.",
                         "Adaptive configuration choice on the same validation; cluster bootstrap does not remove selection bias.",
                         "Five-step histories assume generated recommendations were listened to; their reward is predicted, not observed.",
                         "Top-list reward and full-softmax expected reward are distinct; matched-outcome statistics are selected and not off-policy value estimates.",
                         "No causal satisfaction or retention uplift is measured."],
              "rl_trained": True, "reward_model_retrained": False, "test_evaluated": False}
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "progress.json", {"stage": "complete", "selected_variant": selected, "completed_runs": list(runs)})
    return result


def audit_policy_pilot(output_dir: Path, requested_device: str) -> dict:
    output_dir = output_dir.resolve()
    result = json.loads((output_dir / "result.json").read_text())
    paths = result["paths"]
    context = load_pilot(Path(paths["data_dir"]), Path(paths["reward_data_dir"]), output_dir,
                         Path(paths["config_path"]), Path(paths["protocol_path"]), requested_device, True)
    caches, cache_manifest = prepare_pilot_cache(context)
    if result["provenance"] != context["plan"] or result["cache"] != cache_manifest or result["test_evaluated"]:
        raise ValueError("Pilot report provenance or test flag differs.")
    context["observed_validation"] = load_reward_partition(context["reward_data_dir"], "validation")
    rewards = {split: reward_values(cache["raw_reward"], cache["personal_mean"], context["calibration"], context["duration_scale"])
               for split, cache in caches.items()}
    controls, arrays, baseline_vectors, matched_difference = build_controls(context, caches, rewards)
    if controls != result["controls"] or matched_difference != result["matched_analytic_policy_maximum_difference"]:
        raise ValueError("Pilot controls do not reproduce.")
    if sha256(output_dir / "control-rankings.npz") != result["control_rankings_artifact"]["sha256"]:
        raise ValueError("Pilot control artifact changed.")
    with np.load(output_dir / "control-rankings.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
            raise ValueError("Saved control rankings do not reproduce.")
    forward_differences = {}
    for key, record in result["runs"].items():
        root = output_dir / "runs" / record["variant"]["name"] / f"seed-{record['training_seed']}"
        if json.loads((root / "summary.json").read_text()) != record:
            raise ValueError("Run summary differs from the combined report.")
        audit_policy_run(context, caches["validation"], rewards["validation"], baseline_vectors, root, record)
        policy = new_policy(context)
        policy.load_state_dict(torch.load(root / "policy.pt", map_location="cpu", weights_only=True), strict=True)
        logits = policy_logits(policy, context["validation"], context["device"], context["config"]["execution"]["inference_batch_size"])
        with np.load(root / "validation.npz", allow_pickle=False) as saved:
            difference = float(np.max(np.abs(logits - saved["logits"])))
            if difference > 1e-5 or not np.array_equal(np.argsort(-logits, axis=1, kind="stable")[:, :20] + 3, saved["rankings"]):
                raise ValueError("Validation forward from the saved policy does not reproduce.")
        forward_differences[key] = difference
        print(json.dumps({"stage": "rl_audit_forward_completed", "run": key, "maximum_absolute_difference": difference}), flush=True)
        del policy
        if context["device"].type == "mps":
            torch.mps.empty_cache()
    challenger = select_challenger(context, result["runs"], controls["baseline"])
    name = challenger["variant"]["name"] if challenger else None
    confirmation = [{"training_seed": seed, "eligible": eligible_policy(result["runs"][f"{name}/seed-{seed}"], controls["baseline"], context)}
                    for seed in context["config"]["confirmation_seeds"]] if challenger else []
    selected = name if challenger and all(row["eligible"] for row in confirmation) else "baseline"
    expected_keys = {f"{variant['name']}/seed-{context['config']['training_seed']}" for variant in context["config"]["variants"]}
    if name:
        expected_keys.update(f"{name}/seed-{seed}" for seed in context["config"]["confirmation_seeds"])
    if set(result["runs"]) != expected_keys or name != result["screening_challenger"] or confirmation != result["confirmation"] or selected != result["selected_variant"]:
        raise ValueError("Pilot selection or training budget does not reproduce.")
    checkpoint = context["policy_checkpoint"] if selected == "baseline" else output_dir / "runs" / selected / f"seed-{context['config']['selection']['checkpoint_training_seed']}" / "policy.pt"
    if result["selected_checkpoint"] != {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}:
        raise ValueError("Selected policy checkpoint differs.")
    for modes in result["rollouts"].values():
        for entry in modes.values():
            if sha256(Path(entry["artifact"]["path"])) != entry["artifact"]["sha256"]:
                raise ValueError("Saved rollout artifact changed.")
    if generate_rollouts(context, result["runs"]) != result["rollouts"]:
        raise ValueError("Saved rollout summaries do not reproduce.")
    verify_pilot_inputs(context)
    return {"stage": "rl_pilot_audit_complete", "runs_checked": len(result["runs"]),
            "validation_forward_maximum_differences": forward_differences, "controls_and_selection_recalculated": True,
            "rollout_actions_and_rewards_replayed": True, "frozen_inputs_unchanged": True,
            "selected_variant": selected, "test_evaluated": False}
