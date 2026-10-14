from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recsys_rl.data import load_config, load_partition, sha256, write_json
from recsys_rl.evaluation import evaluate_rankings
from recsys_rl.model import NextItemPolicy, select_device
from recsys_rl.reward import reward_source_hashes
from recsys_rl.reward_catalog import CatalogReward
from recsys_rl.reward_data import load_reward_partition
from recsys_rl.reward_evaluation import control_predictions
from recsys_rl.reward_model import CompletionReward
from recsys_rl.train import batch_tensors


def context_rows(arrays: dict, indices: np.ndarray) -> dict:
    return {name: arrays[name][indices] for name in ("input_ids", "lengths", "user_ids", "timestamps", "history_max_timestamps")}


def new_policy(context: dict) -> NextItemPolicy:
    policy = NextItemPolicy(len(context["catalog"]["items"]), context["policy_config"]["data"]["history_length"],
                            context["policy_config"]["model"])
    policy.load_state_dict(torch.load(context["policy_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    return policy.to(context["device"]).eval()


@torch.no_grad()
def policy_logits(policy: NextItemPolicy, arrays: dict, device: torch.device, batch_size: int) -> np.ndarray:
    policy.eval()
    output = []
    for start in range(0, len(arrays["lengths"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(arrays["lengths"])))
        output.append(policy(*batch_tensors(arrays, indices, device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def verify_pilot_inputs(context: dict) -> None:
    if reward_source_hashes() != context["frozen"]["source_files"]:
        raise ValueError("Policy pilot source changed.")
    for path, expected in context["frozen"]["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Policy pilot input changed: {path}")


def load_pilot(data_dir: Path, reward_data_dir: Path, output_dir: Path, config_path: Path,
               protocol_path: Path, requested_device: str, resume: bool) -> dict:
    data_dir, reward_data_dir, output_dir, config_path, protocol_path = (
        path.resolve() for path in (data_dir, reward_data_dir, output_dir, config_path, protocol_path))
    config = json.loads(config_path.read_text())
    settings = config["training"]
    if config["status"] != "implemented" or settings["method"] != "exact_expected_reward_contextual_bandit":
        raise ValueError("Policy pilot configuration is not implemented.")
    if min(settings["epochs"], settings["batch_size"], settings["learning_rate"], settings["gradient_clip"]) <= 0:
        raise ValueError("Invalid policy pilot training budget.")
    if settings["dropout_enabled"] or settings["checkpoint"] != "final_epoch":
        raise ValueError("Policy pilot requires disabled dropout and final-epoch selection.")
    names = [variant["name"] for variant in config["variants"]]
    if len(names) != 6 or len(set(names)) != 6 or any(variant["kl_beta_base"] < 0 for variant in config["variants"]):
        raise ValueError("Policy pilot requires six unique variants and nonnegative beta.")
    report_paths = {name: (config_path.parent / config[f"{name}_report"]).resolve() for name in ("policy", "reward", "signal")}
    reports = {name: json.loads(path.read_text()) for name, path in report_paths.items()}
    policy_config_path = (config_path.parent / config["policy_config"]).resolve()
    policy_config = load_config(policy_config_path)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    reward_manifest = json.loads((reward_data_dir / "manifest.json").read_text())
    policy_report, reward_report = reports["policy"], reports["reward"]
    if policy_report["provenance"]["frozen"]["preparation_manifest_sha256"] != sha256(data_dir / "manifest.json"):
        raise ValueError("Reference policy used different prepared data.")
    selected = policy_report["runs"][f"{policy_report['selected_variant']}/seed-{policy_report['selected_training_seed']}"]["summary"]
    if sha256(policy_config_path) != selected["config_sha256"] or policy_config["data"] != manifest["config"]["data"]:
        raise ValueError("Reference policy configuration differs.")
    if sha256(reward_data_dir / "manifest.json") != reward_report["preparation_manifest_sha256"] or reward_manifest["policy_manifest_sha256"] != sha256(data_dir / "manifest.json"):
        raise ValueError("Reward preparation differs from policy data.")
    for report in (reward_report, {"source_files_sha256": reports["signal"]["frozen"]["source_files"]}):
        for name, expected in report["source_files_sha256"].items():
            if name != "cli.py" and sha256(Path(__file__).parent / name) != expected:
                raise ValueError(f"Historical computation module changed: {name}")
    checkpoint = Path(policy_report["selected_checkpoint"]["path"]).resolve()
    reward_dir = (config_path.parent / config["reward_output_dir"]).resolve()
    inputs = {str(config_path): sha256(config_path), str(protocol_path): sha256(protocol_path),
              str(policy_config_path): sha256(policy_config_path), str(checkpoint): policy_report["selected_checkpoint"]["sha256"],
              str(data_dir / "manifest.json"): sha256(data_dir / "manifest.json"),
              str(data_dir / "catalog.json"): manifest["catalog_sha256"],
              str(reward_data_dir / "manifest.json"): reward_report["preparation_manifest_sha256"],
              str(Path(reward_manifest["source_path"])): manifest["source"]["sha256"]}
    inputs.update({str(path): sha256(path) for path in report_paths.values()})
    inputs.update({str(data_dir / f"{split}.npz"): value["sha256"] for split, value in manifest["partitions"].items()})
    inputs.update({str(reward_data_dir / f"{split}.npz"): value["sha256"] for split, value in reward_manifest["partitions"].items()})
    inputs.update({str(reward_dir / name): value["sha256"] for name, value in reward_report["artifacts"].items()})
    frozen = {"inputs": inputs, "source_files": reward_source_hashes()}
    plan_path = output_dir / "plan.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Policy pilot output is not empty; use --resume with the same frozen inputs: {output_dir}")
        plan = json.loads(plan_path.read_text())
        if plan["frozen"] != frozen:
            raise ValueError("Policy pilot resume differs from the frozen plan.")
    else:
        plan = {"config": config, "frozen": frozen, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        write_json(plan_path, plan)
    torch.set_num_threads(4)
    device = select_device(requested_device)
    catalog = json.loads((data_dir / "catalog.json").read_text())
    train_all, validation = load_partition(data_dir, "train"), load_partition(data_dir, "validation")
    eligible = np.flatnonzero(train_all["timestamps"] > reward_manifest["cutoffs_seconds"]["reward_train_end"])
    if len(eligible) < config["data"]["train_contexts"]:
        raise ValueError("Not enough late policy-train contexts.")
    indices = np.sort(np.random.default_rng(config["data_seed"]).choice(eligible, config["data"]["train_contexts"], replace=False))
    train = context_rows(train_all, indices)
    reward_train = load_reward_partition(reward_data_dir, "train")
    known_users = np.unique(reward_train["user_ids"])
    counts = np.bincount(reward_train["candidate_tokens"] - 3, minlength=len(catalog["items"]))
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan for item in catalog["items"]])
    known = durations[np.isfinite(durations) & (durations > 0)]
    if not known.size:
        raise ValueError("Time reward needs known train-catalog durations.")
    duration_cap = float(np.quantile(known, config["reward"]["duration_cap_catalog_quantile"]))
    duration_scale = np.minimum(np.where(np.isfinite(durations) & (durations > 0), durations, np.median(known)), duration_cap) / duration_cap
    calibration = reports["signal"]["calibration"]
    scale = calibration["slope"] * calibration["personal_mix_alpha"]
    if calibration["intercept"] < 0 or calibration["intercept"] + calibration["slope"] > 1 or scale <= 0:
        raise ValueError("Matched-scale comparison requires globally affine positive calibration.")
    context = {"config": config, "plan": plan, "frozen": frozen, "output": output_dir, "device": device,
               "paths": {"data_dir": str(data_dir), "reward_data_dir": str(reward_data_dir),
                         "config_path": str(config_path), "protocol_path": str(protocol_path)},
               "policy_config": policy_config, "policy_checkpoint": checkpoint, "reward_dir": reward_dir,
               "reward_report": reward_report, "policy_selected_summary": selected, "manifest": manifest,
               "reward_data_dir": reward_data_dir, "train": train, "validation": validation, "train_indices": indices,
               "eligible_train_queries": len(eligible), "catalog": catalog, "known_users": known_users,
               "counts": counts, "duration_cap": duration_cap, "duration_scale": duration_scale,
               "calibration": calibration, "affine_scale": scale,
               "personal": {split: control_predictions(reward_train, {**arrays, "candidate_tokens": np.full(len(arrays["lengths"]), 3)})["user_mean"]
                            for split, arrays in (("train", train), ("validation", validation))},
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device)}}
    verify_pilot_inputs(context)
    return context


def prepare_pilot_cache(context: dict) -> tuple[dict, dict]:
    root = context["output"] / "cache"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["frozen"] != context["frozen"]:
            raise ValueError("Policy cache inputs differ.")
        for name, value in manifest["artifacts"].items():
            if sha256(root / name) != value["sha256"]:
                raise ValueError(f"Policy cache changed: {name}")
        cached = {}
        for split in ("train", "validation"):
            with np.load(root / f"{split}.npz", allow_pickle=False) as saved:
                cached[split] = dict(saved)
        return cached, manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Incomplete policy cache must be preserved separately before retrying.")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(context["reward_dir"] / "reward.pt", map_location="cpu", weights_only=True)
    reward_model = CompletionReward(context["catalog"], context["policy_config"]["data"]["history_length"], context["known_users"], context["reward_report"]["config"]["model"]).to(context["device"])
    reward_model.load_state_dict(checkpoint["state_dict"], strict=True)
    scorer = CatalogReward(reward_model, context["known_users"], context["device"])
    reference = new_policy(context)
    cached, differences = {}, {}
    for split in ("train", "validation"):
        arrays = context[split]
        print(json.dumps({"stage": "rl_cache_started", "split": split, "contexts": len(arrays["lengths"])}), flush=True)
        raw = scorer.predict(arrays, context["config"]["execution"]["reward_cache_batch_size"])
        logits = policy_logits(reference, arrays, context["device"], context["config"]["execution"]["inference_batch_size"])
        observed = load_reward_partition(context["reward_data_dir"], "selection" if split == "train" else "validation")
        with np.load(context["reward_dir"] / f"{'selection' if split == 'train' else 'validation'}-predictions.npz", allow_pickle=False) as saved:
            expected = saved["model"].copy()
        if split == "train":
            mapping = np.full(context["manifest"]["partitions"]["train"]["queries"], -1, dtype=np.int32)
            mapping[context["train_indices"]] = np.arange(len(context["train_indices"]))
            included = mapping[observed["query_indices"]] >= 0
            rows = mapping[observed["query_indices"][included]]
        else:
            included = np.ones(len(expected), dtype=bool)
            rows = observed["query_indices"]
        difference = float(np.max(np.abs(raw[rows, observed["candidate_tokens"][included] - 3] - expected[included])))
        if difference > context["config"]["execution"]["reward_forward_tolerance"]:
            raise ValueError(f"Catalog reward differs from original forward: {split}/{difference}")
        differences[split] = {"maximum_absolute_difference": difference, "observed_pairs_checked": int(included.sum())}
        cached[split] = {"raw_reward": raw, "reference_logits": logits, "user_ids": arrays["user_ids"], "timestamps": arrays["timestamps"],
                         "original_indices": context["train_indices"] if split == "train" else np.arange(len(arrays["lengths"])),
                         "personal_mean": context["personal"][split]}
        np.savez_compressed(root / f"{split}.npz", **cached[split])
        print(json.dumps({"stage": "rl_cache_completed", "split": split, **differences[split]}), flush=True)
    rankings = np.argsort(-cached["validation"]["reference_logits"], axis=1, kind="stable")[:, :20] + 3
    if evaluate_rankings(context["validation"], rankings, context["catalog"], [5, 10, 20]) != context["policy_selected_summary"]["validation"]["gpt2"]:
        raise ValueError("Reference GPT-2 did not reproduce its historical validation evaluation.")
    generator = np.random.default_rng(context["config"]["rollout"]["sampling_seed"])
    rollout_indices = np.sort(generator.choice(len(context["validation"]["lengths"]), context["config"]["rollout"]["validation_contexts"], replace=False))
    uniforms = generator.random((len(rollout_indices), context["config"]["rollout"]["steps"]))
    np.savez_compressed(root / "rollout-inputs.npz", indices=rollout_indices, uniforms=uniforms)
    manifest = {"frozen": context["frozen"], "reward_forward_checks": differences, "historical_reference_evaluation_reproduced": True,
                "eligible_train_queries": context["eligible_train_queries"], "train_contexts": len(context["train_indices"]),
                "train_timestamp_min": int(context["train"]["timestamps"].min()), "train_timestamp_max": int(context["train"]["timestamps"].max()),
                "duration_cap_seconds": context["duration_cap"], "affine_reward_scale": context["affine_scale"],
                "artifacts": {name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
                              for name in ("train.npz", "validation.npz", "rollout-inputs.npz")}, "test_evaluated": False}
    verify_pilot_inputs(context)
    write_json(manifest_path, manifest)
    return cached, manifest
