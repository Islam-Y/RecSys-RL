from __future__ import annotations

import numpy as np
import torch
from torch import nn

from recsys_rl.evaluation import targets_for
from recsys_rl.policy_evaluation import log_probabilities


class RecencyGate(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        if kind == "global":
            self.bias = nn.Parameter(torch.tensor(-2.0))
            self.network = None
        elif kind == "context":
            self.network = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.constant_(self.network[-1].bias, -2.0)
        else:
            raise ValueError(f"Unknown gate kind: {kind}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.bias.expand(len(features)) if self.network is None else self.network(features).squeeze(-1)
        return torch.sigmoid(logits)


def copy_distribution(arrays: dict, catalog_size: int, tau: float) -> np.ndarray:
    if tau <= 0:
        raise ValueError("Recency decay tau must be positive.")
    positions = np.arange(arrays["input_ids"].shape[1])[None, :]
    tokens = arrays["input_ids"]
    lag = arrays["lengths"][:, None] - 1 - positions
    valid = (positions > 0) & (lag >= 0) & (tokens >= 3) & (tokens < catalog_size+3)
    weights = np.where(valid, np.exp(-np.maximum(lag, 0).astype(np.float64) / tau), 0)
    rows, cols = np.nonzero(valid)
    distribution = np.zeros((len(tokens), catalog_size), dtype=np.float32)
    np.add.at(distribution, (rows, tokens[rows, cols]-3), weights[rows, cols].astype(np.float32))
    sums = distribution.sum(axis=1, keepdims=True)
    return np.divide(distribution, sums, out=distribution, where=sums > 0)


def gate_features(arrays: dict, base: np.ndarray, copy: np.ndarray) -> np.ndarray:
    n, catalog_size = base.shape
    features = np.empty((n, 8), dtype=np.float32)
    positions = np.arange(arrays["input_ids"].shape[1])[None, :]
    lag = arrays["lengths"][:, None] - 1 - positions
    known = (positions > 0) & (lag >= 0) & (arrays["input_ids"] >= 3)
    counts = known.sum(axis=1)
    unique = (copy > 0).sum(axis=1)
    length = arrays["lengths"].astype(np.float32)-1
    features[:, 0] = counts / np.maximum(length, 1)
    features[:, 1] = unique / np.maximum(counts, 1)
    features[:, 2] = length / (arrays["input_ids"].shape[1]-1)
    features[:, 3] = -np.sum(copy * np.log(np.maximum(copy, 1e-38)), axis=1) / np.log(catalog_size)
    features[:, 4] = -np.sum(base * np.log(np.maximum(base, 1e-38)), axis=1) / np.log(catalog_size)
    features[:, 5] = np.sum(base * (copy > 0), axis=1)
    features[:, 6] = np.sum(base * copy, axis=1)
    features[:, 7] = np.where(counts > 0, np.min(np.where(known, lag, arrays["input_ids"].shape[1]-1), axis=1), arrays["input_ids"].shape[1]-1) / (arrays["input_ids"].shape[1]-1)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite gate features.")
    return features


def batch_rows(arrays: dict, start: int, stop: int) -> dict:
    return {key: arrays[key][start:stop] for key in ("input_ids", "lengths")}


def training_inputs(arrays: dict, logits: np.ndarray, tau: float, batch_size: int = 512) -> dict:
    targets = [targets_for(arrays, i)-3 for i in range(len(arrays["lengths"]))]
    maximum = max(map(len, targets))
    labels = np.zeros((len(targets), maximum), dtype=np.int64)
    mask = np.zeros_like(labels, dtype=np.float32)
    for i, row in enumerate(targets):
        labels[i, :len(row)] = row
        mask[i, :len(row)] = 1
    features, base_targets, copy_targets, available = [], [], [], []
    for start in range(0, len(targets), batch_size):
        stop = min(start+batch_size, len(targets))
        rows = batch_rows(arrays, start, stop)
        base = np.exp(log_probabilities(logits[start:stop])).astype(np.float32)
        copy = copy_distribution(rows, logits.shape[1], tau)
        features.append(gate_features(rows, base, copy))
        base_targets.append(np.take_along_axis(base, labels[start:stop], axis=1))
        copy_targets.append(np.take_along_axis(copy, labels[start:stop], axis=1))
        available.append((copy.sum(axis=1) > 0).astype(np.float32))
    return {"features": np.concatenate(features), "base": np.concatenate(base_targets),
            "copy": np.concatenate(copy_targets), "available": np.concatenate(available), "mask": mask}


@torch.no_grad()
def mixture_predictions(gate: RecencyGate, arrays: dict, logits: np.ndarray, tau: float,
                        batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate.eval()
    rankings, all_gates, target_losses = [], [], []
    for start in range(0, len(arrays["lengths"]), batch_size):
        stop = min(start+batch_size, len(arrays["lengths"]))
        base = np.exp(log_probabilities(logits[start:stop])).astype(np.float32)
        copy = copy_distribution(batch_rows(arrays, start, stop), logits.shape[1], tau)
        features = gate_features(batch_rows(arrays, start, stop), base, copy)
        gates = gate(torch.from_numpy(features)).numpy() * (copy.sum(axis=1) > 0)
        probabilities = (1-gates[:, None])*base + gates[:, None]*copy
        if np.any(probabilities < 0) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Invalid recency mixture distribution.")
        rankings.append((np.argsort(-probabilities, axis=1, kind="stable")[:, :20]+3).astype(np.int32))
        all_gates.append(gates)
        if "target_tokens" in arrays:
            for local, index in enumerate(range(start, stop)):
                target_losses.append(float(-np.log(np.maximum(probabilities[local, targets_for(arrays, index)-3], 1e-38)).mean()))
    return np.concatenate(rankings), np.concatenate(all_gates), np.asarray(target_losses)


@torch.no_grad()
def mixture_probabilities(gate: RecencyGate, arrays: dict, logits: np.ndarray, tau: float) -> np.ndarray:
    base = np.exp(log_probabilities(logits)).astype(np.float32)
    copy = copy_distribution(arrays, logits.shape[1], tau)
    values = gate(torch.from_numpy(gate_features(arrays, base, copy))).numpy() * (copy.sum(axis=1) > 0)
    return (1-values[:, None])*base + values[:, None]*copy
