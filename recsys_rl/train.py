from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn import functional as functional

from recsys_rl.data import load_config, load_partition, sha256, write_json
from recsys_rl.evaluation import control_rankings, evaluate_rankings, targets_for
from recsys_rl.model import NextItemPolicy, seed_everything, select_device


def batch_tensors(arrays: dict, indices: np.ndarray, device: torch.device):
    lengths = torch.as_tensor(arrays["lengths"][indices].astype(np.int64), device=device)
    maximum = int(arrays["lengths"][indices].max())
    inputs = torch.as_tensor(arrays["input_ids"][indices, :maximum].astype(np.int64), device=device)
    return inputs, lengths


@torch.no_grad()
def predict(model: NextItemPolicy, arrays: dict, device: torch.device, batch_size: int, top_n: int) -> np.ndarray:
    model.eval()
    rankings = []
    for offset in range(0, len(arrays["lengths"]), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(arrays["lengths"])))
        inputs, lengths = batch_tensors(arrays, indices, device)
        scores = model(inputs, lengths).cpu().numpy()
        rankings.append(np.argsort(-scores, axis=1, kind="stable")[:, :top_n] + 3)
    return np.concatenate(rankings).astype(np.int32)


def train(data_dir: Path, output_dir: Path, config_path: Path, requested_device: str, protocol_path: Path | None = None) -> dict:
    config = load_config(config_path)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    prepared_config = manifest["config"]
    if config["data"] != prepared_config["data"] or config["seed"] != prepared_config["seed"]:
        raise ValueError("Data settings or sampling seed differ from the preparation; prepare separate data.")
    config_hash = sha256(config_path)
    protocol_hash = sha256(protocol_path) if protocol_path is not None else manifest["protocol_sha256"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    settings = config["training"]
    training_seed = settings.get("seed", config["seed"])
    seed_everything(training_seed)
    device = select_device(requested_device)
    catalog = json.loads((data_dir / "catalog.json").read_text())
    if sha256(data_dir / "catalog.json") != manifest["catalog_sha256"]:
        raise ValueError("Catalog hash differs from the frozen preparation.")
    train_rows = load_partition(data_dir, "train")
    validation = load_partition(data_dir, "validation")
    model = NextItemPolicy(len(catalog["items"]), config["data"]["history_length"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    top_k = config["evaluation"]["top_k"]
    if max(top_k) > len(catalog["items"]):
        raise ValueError("Ranking cutoff exceeds the prepared catalog size.")
    controls = {name: evaluate_rankings(validation, ranking, catalog, top_k) for name, ranking in control_rankings(validation, catalog, max(top_k)).items()}
    history = []
    best = -float("inf")
    patience = settings.get("early_stopping_patience")
    epochs_without_improvement = 0
    stop_reason = "epoch_budget_exhausted"
    started = time.perf_counter()
    print(json.dumps({"stage": "training_started", "device": str(device), "training_seed": training_seed, "parameters": sum(p.numel() for p in model.parameters()), "train_queries": len(train_rows["lengths"]), "validation_queries": len(validation["lengths"])}), flush=True)
    for epoch in range(settings["epochs"]):
        model.train()
        order = np.random.default_rng(training_seed + epoch).permutation(len(train_rows["lengths"]))
        losses = []
        epoch_started = time.perf_counter()
        for offset in range(0, len(order), settings["batch_size"]):
            indices = order[offset:offset + settings["batch_size"]]
            inputs, lengths = batch_tensors(train_rows, indices, device)
            targets = [targets_for(train_rows, int(index)) - 3 for index in indices]
            maximum = max(map(len, targets))
            target_array = np.full((len(indices), maximum), -100, dtype=np.int64)
            for index, values in enumerate(targets):
                target_array[index, :len(values)] = values
            labels = torch.as_tensor(target_array, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            log_probs = functional.log_softmax(logits, dim=-1)
            valid = labels != -100
            per_target = -log_probs.gather(1, labels.clamp_min(0)) * valid
            loss = (per_target.sum(dim=1) / valid.sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite next-item training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            losses.append((float(loss.detach().cpu()), len(indices)))
            if offset % (settings["batch_size"] * 100) == 0:
                print(json.dumps({"stage": "train_batch", "epoch": epoch + 1, "queries_seen": offset + len(indices), "loss": losses[-1][0]}), flush=True)
        rankings = predict(model, validation, device, settings["batch_size"], max(top_k))
        evaluation = evaluate_rankings(validation, rankings, catalog, top_k)
        record = {"epoch": epoch + 1, "train_loss": sum(loss * count for loss, count in losses) / sum(count for _, count in losses), "seconds": time.perf_counter() - epoch_started, "validation": evaluation}
        history.append(record)
        metric = evaluation["metrics"][config["evaluation"]["checkpoint_metric"]]
        if metric > best:
            best = metric
            epochs_without_improvement = 0
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, output_dir / "baseline.pt")
            np.savez_compressed(output_dir / "validation-top-items.npz", rankings=rankings, user_ids=validation["user_ids"], timestamps=validation["timestamps"], target_tokens=validation["target_tokens"], target_offsets=validation["target_offsets"])
            best_epoch = epoch + 1
            best_result = evaluation
        else:
            epochs_without_improvement += 1
        write_json(output_dir / "training-history.json", {"epochs": history})
        print(json.dumps({"stage": "epoch_completed", "epoch": epoch + 1, "train_loss": record["train_loss"], "validation_ndcg_at_10": metric, "seconds": record["seconds"]}), flush=True)
        if patience is not None and epochs_without_improvement >= patience:
            stop_reason = "early_stopping_patience_exhausted"
            print(json.dumps({"stage": "early_stopping", "epoch": epoch + 1, "selected_epoch": best_epoch, "patience": patience}), flush=True)
            break
    result = {"experiment": config.get("experiment", "sequential-baseline-v1"), "config": config, "config_sha256": config_hash, "protocol_sha256": protocol_hash,
        "preparation_config_sha256": manifest["config_sha256"], "preparation_protocol_sha256": manifest["protocol_sha256"], "preparation_manifest_sha256": sha256(data_dir / "manifest.json"),
        "data": {key: manifest[key] for key in ("source", "cutoffs_seconds", "population_users", "catalog_size", "partitions", "ranking_scope")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "device": str(device), "determinism": "Fixed RNG seeds; exact cross-device floating-point reproducibility is not guaranteed."},
        "parameters": sum(p.numel() for p in model.parameters()), "total_seconds": time.perf_counter() - started,
        "data_seed": config["seed"], "training_seed": training_seed,
        "epochs_completed": len(history), "early_stopping_patience": patience, "stop_reason": stop_reason,
        "selected_epoch": best_epoch, "validation": {"gpt2": best_result, **controls}, "training_history": history,
        "model_sha256": sha256(output_dir / "baseline.pt"), "test_evaluated": False, "reward_model_trained": False, "rl_trained": False}
    write_json(output_dir / "summary.json", result)
    return result
