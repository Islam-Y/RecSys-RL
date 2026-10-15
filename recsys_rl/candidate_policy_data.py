from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recsys_rl.data import load_partition, sha256, write_json
from recsys_rl.history_reward_data import connection, load_npz
from recsys_rl.model import select_device
from recsys_rl.policy_data import context_rows, new_policy, policy_logits
from recsys_rl.ranked_policy_data import catalog_reward
from recsys_rl.ranked_policy_experiment import reward_regression
from recsys_rl.ranking_data import choose_users, queries_for
from recsys_rl.recency_data import read_json
from recsys_rl.reward import reward_source_hashes


def verify(context: dict) -> None:
    frozen = context["plan"]["frozen"]
    if reward_source_hashes() != frozen["source_files"]:
        raise ValueError("Candidate-policy source changed.")
    for path, expected in frozen["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Candidate-policy input changed: {path}")


def load_context(config_path: Path, protocol: Path, output: Path, device: str, resume: bool) -> dict:
    config_path, protocol, output = [p.resolve() for p in (config_path, protocol, output)]
    config = read_json(config_path)
    resolve = lambda key: (config_path.parent / config[key]).resolve()
    historical = read_json(resolve("historical_report"))
    execution = read_json(resolve("historical_execution"))
    if sha256(resolve("historical_report")) != execution["result_sha256"]:
        raise ValueError("Historical policy report changed.")
    frozen = historical["provenance"]["frozen"]
    for name, expected in frozen["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical computation changed: {name}")
    inputs = dict(frozen["inputs"])
    old_run = historical["runs"][f"{historical['selection']['selected_variant']}/42"]
    checkpoint = Path(old_run["checkpoint"])
    if sha256(checkpoint) != old_run["artifacts"][checkpoint.name]:
        raise ValueError("Historical policy checkpoint changed.")
    for p in (config_path, protocol, resolve("historical_report"), resolve("historical_execution"), checkpoint):
        inputs[str(p)] = sha256(p)
    own = {"inputs": inputs, "source_files": reward_source_hashes()}
    paths = {"config": str(config_path), "protocol": str(protocol), "output": str(output), "data": str(resolve("new_data_dir"))}
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError("Candidate-policy output exists; use --resume or --audit.")
        plan = read_json(output / "plan.json")
        if plan["frozen"] != own or plan["paths"] != paths:
            raise ValueError("Candidate-policy plan changed.")
    else:
        plan = {"config": config, "frozen": own, "paths": paths,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "plan.json", plan)
    data_dir = next(Path(p).parent for p in inputs if p.endswith("data/processed/mvp/manifest.json"))
    reward_dir = next(Path(p).parent for p in inputs if p.endswith("data/processed/reward-v1/manifest.json"))
    ranking = read_json(config_path.parent.parent / "experiments/ranking-result.json")
    ranker_run = ranking["runs"][f"{ranking['selection']['selected_variant']}/42"]
    gate = read_json(resolve("recency_execution"))["selected_global_gate"]
    original = load_partition(data_dir, "train")
    lower = read_json(reward_dir / "manifest.json")["cutoffs_seconds"]["reward_train_end"]
    expanded = np.flatnonzero(original["timestamps"] > lower)
    pilot = next(Path(p) for p in inputs if p.endswith("policy-rl-pilot-v1/cache/train.npz"))
    small = load_npz(pilot)["original_indices"]
    positions = np.searchsorted(expanded, small)
    if np.any(positions >= len(expanded)) or not np.array_equal(expanded[positions], small):
        raise ValueError("Small training contexts are not a subset of expanded.")
    context = {"config": config, "plan": plan, "output": output, "new_data_dir": resolve("new_data_dir"),
               "data_dir": data_dir, "manifest": read_json(data_dir / "manifest.json"),
               "catalog": read_json(data_dir / "catalog.json"),
               "policy_config": read_json(config_path.parent / "baseline-selected.json"),
               "policy_checkpoint": next(Path(p) for p in inputs if p.endswith("small-tuning-v1/runs/lr-low/seed-42/baseline.pt")),
               "historical_checkpoint": checkpoint, "device": select_device(device),
               "source": next(Path(p) for p in inputs if p.endswith("yambda-50m-multi_event.parquet")),
               "ranker_run": ranker_run, "calibration": ranking["calibration"],
               "priors": load_npz(resolve("ranking_data_dir") / "priors.npz"),
               "previous_populations": load_npz(resolve("ranking_data_dir") / "populations.npz"),
               "gate": gate["sigmoid_bias"], "tau": gate["variant"]["tau"],
               "train": context_rows(original, expanded), "train_indices": expanded,
               "small_positions": positions, "training_lower_bound": lower}
    torch.set_num_threads(config["execution"]["threads"])
    verify(context)
    return context


def build_groups(context: dict) -> dict:
    settings = context["config"]["data"]
    previous = context["previous_populations"]
    excluded = np.union1d(previous["confirmation_excluded_users"], previous["confirmation_users"])
    cuts = context["manifest"]["cutoffs_seconds"]
    db = connection(context)
    try:
        development = choose_users(db, excluded, settings["development_users"], settings["development_user_seed"], cuts["train_end"])
        final_excluded = np.union1d(excluded, development)
        final = choose_users(db, final_excluded, settings["final_users"], settings["final_user_seed"], cuts["train_end"])
        arrays = {
            "validation": queries_for(context, db, development, cuts["train_end"], cuts["validation_end"], settings["development_query_seed"], limit=settings["development_queries"]),
            "final": queries_for(context, db, final, cuts["validation_end"], cuts["data_end"], settings["final_query_seed"], limit=settings["final_queries"]),
            "populations": {"excluded_users": excluded, "development_users": development, "final_users": final},
            "train": {**context["train"], "original_indices": context["train_indices"], "small_positions": context["small_positions"]}}
    finally:
        db.close()
    if np.intersect1d(np.unique(context["train"]["user_ids"]), np.union1d(development, final)).size:
        raise ValueError("New cohorts overlap policy training.")
    return arrays


def prepare(context: dict) -> dict:
    root = context["new_data_dir"]
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if manifest["frozen"] != context["plan"]["frozen"]:
            raise ValueError("Candidate-policy preparation changed.")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Candidate-policy prepared artifact changed.")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Preserve incomplete candidate-policy preparation.")
        arrays = build_groups(context)
        root.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            np.savez_compressed(root / f"{name}.npz", **value)
        manifest = {"frozen": context["plan"]["frozen"], "source_commit": context["plan"]["source_commit"],
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "partitions": {name: {"queries": len(arrays[name]["lengths"]), "query_users": len(np.unique(arrays[name]["user_ids"])), "targets": len(arrays[name]["target_tokens"])} for name in ("validation", "final")},
                    "populations": {name: len(v) for name, v in arrays["populations"].items()},
                    "small_train_contexts": len(context["small_positions"]), "expanded_train_contexts": len(context["train_indices"]),
                    "train_timestamp_min": int(context["train"]["timestamps"].min()), "training_lower_bound": context["training_lower_bound"],
                    "final_response_read": False, "final_predictions_read": False,
                    "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}}
        write_json(root / "manifest.json", manifest)
    context["validation"] = load_npz(root / "validation.npz")
    verify(context)
    return manifest


def prepare_cache(context: dict) -> dict:
    root = context["output"] / "cache"
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected:
                raise ValueError("Candidate-policy cache changed.")
        return {split: load_npz(root / f"{split}.npz") for split in ("train", "validation")}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve incomplete candidate-policy cache.")
    root.mkdir()
    policy = new_policy(context)
    caches = {}
    for split in ("train", "validation"):
        caches[split] = {"reward": catalog_reward(context, context[split]),
                         "reference_logits": policy_logits(policy, context[split], context["device"], context["config"]["execution"]["inference_batch_size"])}
        np.savez_compressed(root / f"{split}.npz", **caches[split])
    regression, observed = reward_regression(context, context["validation"], caches["validation"]["reward"])
    np.savez_compressed(root / "observed-validation.npz", **observed)
    write_json(root / "manifest.json", {"development_reward_model": regression, "final_opened": False,
                                       "artifacts": {p.name: sha256(p) for p in sorted(root.glob("*.npz"))}})
    verify(context)
    return caches


def training_context(context: dict, caches: dict, variant: dict) -> tuple[dict, dict]:
    if variant["data"] == "expanded":
        return context, caches
    if variant["data"] != "small":
        raise ValueError("Unknown training population.")
    indices = context["small_positions"]
    return ({**context, "train": {k: v[indices] for k, v in context["train"].items()}},
            {**caches, "train": {k: v[indices] for k, v in caches["train"].items()}})
