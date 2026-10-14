from __future__ import annotations

import numpy as np
import torch
from torch import nn


def user_tokens(user_ids: np.ndarray, known_users: np.ndarray) -> np.ndarray:
    if not known_users.size:
        raise ValueError("Reward training user vocabulary is empty.")
    positions = np.searchsorted(known_users, user_ids)
    matched = (positions < len(known_users)) & (known_users[np.minimum(positions, len(known_users) - 1)] == user_ids)
    return np.where(matched, positions + 1, 0).astype(np.int64)


def duration_features(catalog: dict) -> np.ndarray:
    durations = np.asarray([item["duration_seconds"] if item["duration_seconds"] is not None else np.nan
                            for item in catalog["items"]], dtype=np.float64)
    known = np.isfinite(durations) & (durations > 0)
    transformed = np.log1p(durations[known])
    center = float(transformed.mean()) if transformed.size else 0.0
    scale = float(transformed.std()) if transformed.size else 1.0
    features = np.zeros((len(durations) + 3, 2), dtype=np.float32)
    features[3:, 1] = known
    features[np.flatnonzero(known) + 3, 0] = np.clip((transformed - center) / max(scale, 1e-6), -3, 3)
    return features


class CompletionReward(nn.Module):
    def __init__(self, catalog: dict, history_length: int, known_users: np.ndarray, settings: dict):
        super().__init__()
        width = settings["embedding_size"]
        self.history_length = history_length
        self.item_embedding = nn.Embedding(len(catalog["items"]) + 3, width, padding_idx=0)
        self.user_embedding = nn.Embedding(len(known_users) + 1, settings["user_embedding_size"], padding_idx=0)
        self.register_buffer("duration_features", torch.from_numpy(duration_features(catalog)))
        self.register_buffer("known_users", torch.as_tensor(known_users.astype(np.int64)))
        first, second = settings["hidden_sizes"]
        self.head = nn.Sequential(nn.Linear(4 * width + settings["user_embedding_size"] + 5, first),
                                  nn.GELU(), nn.Dropout(settings["dropout"]), nn.Linear(first, second),
                                  nn.GELU(), nn.Linear(second, 1))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, candidates: torch.Tensor,
                users: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None, :]
        history_mask = (positions > 0) & (positions < lengths[:, None])
        count = (lengths - 1).to(torch.float32)
        embedded = self.item_embedding(inputs)
        history = (embedded * history_mask[:, :, None]).sum(dim=1) / count[:, None]
        last = embedded[torch.arange(len(lengths), device=inputs.device), lengths - 1]
        candidate = self.item_embedding(candidates)
        repeats = ((inputs == candidates[:, None]) & history_mask).any(dim=1).to(torch.float32)
        unknown_fraction = ((inputs == 2) & history_mask).sum(dim=1) / count
        metadata = torch.cat((self.duration_features[candidates], repeats[:, None],
                              (count / self.history_length)[:, None], unknown_fraction[:, None]), dim=1)
        features = torch.cat((history, last, candidate, history * candidate, self.user_embedding(users), metadata), dim=1)
        return torch.sigmoid(self.head(features).squeeze(1))


def reward_batch(arrays: dict, indices: np.ndarray, known_users: np.ndarray, device: torch.device) -> tuple:
    maximum = int(arrays["lengths"][indices].max())
    return (torch.as_tensor(arrays["input_ids"][indices, :maximum].astype(np.int64), device=device),
            torch.as_tensor(arrays["lengths"][indices].astype(np.int64), device=device),
            torch.as_tensor(arrays["candidate_tokens"][indices].astype(np.int64), device=device),
            torch.as_tensor(user_tokens(arrays["user_ids"][indices], known_users), device=device))
