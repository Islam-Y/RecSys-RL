from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from recsys_rl.data import sha256, write_json
from recsys_rl.model import seed_everything, select_device
from recsys_rl.reward_data import load_reward_config, load_reward_partition
from recsys_rl.reward_evaluation import control_predictions, evaluate_reward, regression_metrics
from recsys_rl.reward_model import CompletionReward, reward_batch


def reward_source_hashes() -> dict:
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def verify_reward_inputs(data_dir: Path, manifest_hash: str, sources: dict) -> None:
    if sha256(data_dir / "manifest.json") != manifest_hash or reward_source_hashes() != sources:
        raise ValueError("Reward preparation or experiment code changed.")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    parent_dir = Path(manifest["policy_data_dir"])
    if sha256(parent_dir / "manifest.json") != manifest["policy_manifest_sha256"]:
        raise ValueError("Original policy manifest changed.")
    if sha256(parent_dir / "catalog.json") != manifest["policy_catalog_sha256"]:
        raise ValueError("Original policy catalog changed.")
    for split, expected in manifest["policy_partitions_sha256"].items():
        if sha256(parent_dir / f"{split}.npz") != expected:
            raise ValueError(f"Original policy partition changed: {split}")
    for split, values in manifest["partitions"].items():
        if sha256(data_dir / f"{split}.npz") != values["sha256"]:
            raise ValueError(f"Reward partition changed: {split}")
    if sha256(Path(manifest["source_path"])) != manifest["source"]["sha256"]:
        raise ValueError("Raw reward source changed.")
    checkpoint = manifest["reference_policy_checkpoint"]
    if Path(checkpoint["path"]).exists() and sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("Reference policy checkpoint changed.")


@torch.no_grad()
def predict_reward(model: CompletionReward, arrays: dict, known_users: np.ndarray,
                   device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    predictions = []
    for offset in range(0, len(arrays["rewards"]), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(arrays["rewards"])))
        predictions.append(model(*reward_batch(arrays, indices, known_users, device)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def save_predictions(path: Path, arrays: dict, predictions: dict) -> None:
    fields = {name: arrays[name] for name in ("user_ids", "timestamps", "candidate_tokens", "query_indices", "rewards")}
    np.savez_compressed(path, **fields, **predictions)


def audit_reward_run(data_dir: Path, output_dir: Path) -> dict:
    result = json.loads((output_dir / "summary.json").read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    verify_reward_inputs(data_dir, result["preparation_manifest_sha256"], result["source_files_sha256"])
    if any(result[key] for key in ("test_evaluated", "test_labels_read", "rl_trained")):
        raise ValueError("Unexpected experiment stage flags.")
    if sha256(Path(result["config_path"])) != result["config_sha256"] or sha256(Path(result["protocol_path"])) != result["protocol_sha256"]:
        raise ValueError("Reward configuration or protocol changed.")
    for name, values in result["artifacts"].items():
        if sha256(output_dir / name) != values["sha256"]:
            raise ValueError(f"Reward artifact changed: {name}")
    train = load_reward_partition(data_dir, "train")
    known_users, known_items = np.unique(train["user_ids"]), np.unique(train["candidate_tokens"])
    catalog = json.loads((Path(manifest["policy_data_dir"]) / "catalog.json").read_text())
    for split in ("selection", "validation"):
        arrays = load_reward_partition(data_dir, split)
        controls = control_predictions(train, arrays)
        with np.load(output_dir / f"{split}-predictions.npz", allow_pickle=False) as saved:
            for name in ("user_ids", "timestamps", "candidate_tokens", "query_indices", "rewards"):
                if not np.array_equal(saved[name], arrays[name]):
                    raise ValueError(f"Saved reward population changed: {split}/{name}")
            for name, expected in controls.items():
                if not np.array_equal(saved[name], expected):
                    raise ValueError(f"Reward control changed: {split}/{name}")
            for name in ("model", *controls):
                measured = evaluate_reward(arrays, saved[name], catalog, known_users, known_items,
                                           result["config"]["evaluation"]["calibration_bins"])
                if measured != result[split][name]:
                    raise ValueError(f"Reward metrics differ: {split}/{name}")
    history = json.loads((output_dir / "training-history.json").read_text())["epochs"]
    selected = min(history, key=lambda row: row["selection"]["mse"])
    if selected["epoch"] != result["selected_epoch"] or selected["selection"] != result["selection"]["model"]["metrics"]:
        raise ValueError("Reward checkpoint selection differs from the protocol.")
    if history != result["training_history"]:
        raise ValueError("Reward training history changed.")
    return {"regression_metrics_recalculated": True, "controls_recalculated": True,
            "saved_populations_match": True, "selected_epoch_checked": True,
            "source_config_protocol_data_and_artifact_hashes_checked": True, "test_evaluated": False}


def train_reward(data_dir: Path, output_dir: Path, config_path: Path, protocol_path: Path, requested_device: str) -> dict:
    config = load_reward_config(config_path)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    manifest_hash, sources = sha256(data_dir / "manifest.json"), reward_source_hashes()
    if config != manifest["config"] or sha256(config_path) != manifest["config_sha256"] or sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Reward configuration or protocol differs from preparation.")
    reference_path = config_path.parent / config["reference_policy_report"]
    if sha256(reference_path) != manifest["reference_policy_report_sha256"]:
        raise ValueError("Reference policy report changed.")
    verify_reward_inputs(data_dir, manifest_hash, sources)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Reward training output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    torch.set_num_threads(4)
    seed_everything(config["seed"])
    device = select_device(requested_device)
    settings = config["training"]
    train, selection = load_reward_partition(data_dir, "train"), load_reward_partition(data_dir, "selection")
    known_users, known_items = np.unique(train["user_ids"]), np.unique(train["candidate_tokens"])
    catalog = json.loads((Path(manifest["policy_data_dir"]) / "catalog.json").read_text())
    model = CompletionReward(catalog, manifest["history_length"], known_users, config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    controls = control_predictions(train, selection)
    history, best, without_improvement = [], float("inf"), 0
    started = time.perf_counter()
    stop_reason = "epoch_budget_exhausted"
    print(json.dumps({"stage": "reward_training_started", "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "train_pairs": len(train["rewards"]), "selection_pairs": len(selection["rewards"])}), flush=True)
    for epoch in range(settings["epochs"]):
        model.train()
        order = np.random.default_rng(config["seed"] + epoch).permutation(len(train["rewards"]))
        total_loss = 0.0
        epoch_started = time.perf_counter()
        for offset in range(0, len(order), settings["batch_size"]):
            indices = order[offset:offset + settings["batch_size"]]
            labels = torch.as_tensor(train["rewards"][indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*reward_batch(train, indices, known_users, device))
            loss = functional.mse_loss(prediction, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite reward loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        predictions = predict_reward(model, selection, known_users, device, settings["batch_size"])
        measured = regression_metrics(selection["rewards"], predictions)
        record = {"epoch": epoch + 1, "train_mse": total_loss / len(order), "selection": measured,
                  "seconds": time.perf_counter() - epoch_started}
        history.append(record)
        if measured["mse"] < best:
            best, without_improvement, best_epoch = measured["mse"], 0, epoch + 1
            torch.save({"epoch": best_epoch, "config": config,
                        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, output_dir / "reward.pt")
            save_predictions(output_dir / "selection-predictions.npz", selection, {"model": predictions, **controls})
        else:
            without_improvement += 1
        write_json(output_dir / "training-history.json", {"epochs": history})
        print(json.dumps({"stage": "reward_epoch_completed", "epoch": epoch + 1, "train_mse": record["train_mse"],
                          "selection_mse": measured["mse"], "seconds": record["seconds"]}), flush=True)
        if without_improvement >= settings["early_stopping_patience"]:
            stop_reason = "early_stopping_patience_exhausted"
            break
    checkpoint = torch.load(output_dir / "reward.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != best_epoch or checkpoint["config"] != config:
        raise ValueError("Saved reward checkpoint metadata differs.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    validation = load_reward_partition(data_dir, "validation")
    validation_predictions = {"model": predict_reward(model, validation, known_users, device, settings["batch_size"]),
                              **control_predictions(train, validation)}
    save_predictions(output_dir / "validation-predictions.npz", validation, validation_predictions)
    evaluations = {}
    for split, arrays in (("selection", selection), ("validation", validation)):
        with np.load(output_dir / f"{split}-predictions.npz", allow_pickle=False) as saved:
            evaluations[split] = {name: evaluate_reward(arrays, saved[name], catalog, known_users, known_items,
                                                       config["evaluation"]["calibration_bins"])
                                  for name in ("model", "global_mean", "item_mean", "user_mean")}
    verify_reward_inputs(data_dir, manifest_hash, sources)
    if sha256(config_path) != manifest["config_sha256"] or sha256(protocol_path) != manifest["protocol_sha256"] or sha256(reference_path) != manifest["reference_policy_report_sha256"]:
        raise ValueError("Reward protocol, configuration or reference changed during training.")
    result = {"experiment": config["experiment"], "source_commit": commit, "source_files_sha256": sources,
              "config": config, "config_path": str(config_path), "config_sha256": sha256(config_path),
              "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
              "preparation_manifest_sha256": manifest_hash, "data": manifest,
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                              "numpy": np.__version__, "device": str(device), "determinism": "Fixed RNG seed; exact cross-device floating-point reproducibility is not guaranteed."},
              "parameters": sum(p.numel() for p in model.parameters()), "total_seconds": time.perf_counter() - started,
              "epochs_completed": len(history), "selected_epoch": best_epoch, "stop_reason": stop_reason,
              "training_history": history, **evaluations,
              "artifacts": {name: {"sha256": sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
                            for name in ("reward.pt", "selection-predictions.npz", "validation-predictions.npz", "training-history.json")},
              "test_labels_read": False, "test_evaluated": False, "reward_model_trained": True, "rl_trained": False,
              "limits": ["Observed listens only: completion conditional on listening, not click or exposure response probability.",
                         "One fixed recipe and training seed; selection controls early stopping, policy validation is evaluated only after checkpoint selection.",
                         "Same restricted catalog and substantial UNK history; catalog metadata comes from the full original policy train period.",
                         "Mean history embeddings plus the last item do not encode the full event order.",
                         "User identity is used; unseen users use a zero embedding and must be inspected separately.",
                         "No causal user satisfaction, retention or unseen-candidate accuracy claim; organic and recommendation-driven events are mixed in training."]}
    write_json(output_dir / "summary.json", result)
    result["audit"] = audit_reward_run(data_dir, output_dir)
    write_json(output_dir / "summary.json", result)
    return result
