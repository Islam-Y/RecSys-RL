from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional

from recsys_rl.reward_model import CompletionReward, user_tokens


class CatalogReward:
    """Evaluate the frozen reward head while sharing history work across candidates."""

    def __init__(self, model: CompletionReward, known_users: np.ndarray, device: torch.device):
        self.model, self.known_users, self.device = model, known_users, device
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        width = model.item_embedding.embedding_dim
        self.width = width
        self.user_start = 4 * width
        self.metadata_start = self.user_start + model.user_embedding.embedding_dim
        weight = model.head[0].weight
        self.items = model.item_embedding.weight[3:]
        self.candidate_offset = functional.linear(self.items, weight[:, 2 * width:3 * width])
        self.candidate_offset = self.candidate_offset + functional.linear(model.duration_features[3:], weight[:, self.metadata_start:self.metadata_start + 2])

    @torch.no_grad()
    def score_batch(self, arrays: dict, indices: np.ndarray) -> torch.Tensor:
        model, width = self.model, self.width
        lengths = torch.as_tensor(arrays["lengths"][indices].astype(np.int64), device=self.device)
        maximum = int(arrays["lengths"][indices].max())
        inputs = torch.as_tensor(arrays["input_ids"][indices, :maximum].astype(np.int64), device=self.device)
        users = torch.as_tensor(user_tokens(arrays["user_ids"][indices], self.known_users), device=self.device)
        positions = torch.arange(maximum, device=self.device)[None, :]
        mask = (positions > 0) & (positions < lengths[:, None])
        count = (lengths - 1).to(torch.float32)
        embedded = model.item_embedding(inputs)
        history = (embedded * mask[:, :, None]).sum(dim=1) / count[:, None]
        last = embedded[torch.arange(len(indices), device=self.device), lengths - 1]
        unknown = ((inputs == 2) & mask).sum(dim=1) / count
        weight, bias = model.head[0].weight, model.head[0].bias
        common = functional.linear(history, weight[:, :width], bias)
        common = common + functional.linear(last, weight[:, width:2 * width])
        common = common + functional.linear(model.user_embedding(users), weight[:, self.user_start:self.metadata_start])
        common = common + functional.linear(torch.stack((count / model.history_length, unknown), dim=1), weight[:, self.metadata_start + 3:])
        interaction = torch.matmul(history[:, None, :] * weight[None, :, 3 * width:4 * width], self.items.T).transpose(1, 2)
        hidden = interaction + common[:, None, :] + self.candidate_offset[None, :, :]
        occurrences = torch.zeros((len(indices), model.item_embedding.num_embeddings), device=self.device)
        occurrences.scatter_add_(1, inputs, mask.to(torch.float32))
        hidden = hidden + (occurrences[:, 3:] > 0)[:, :, None] * weight[None, None, :, self.metadata_start + 2]
        for layer in list(model.head.children())[1:]:
            hidden = layer(hidden)
        return torch.sigmoid(hidden.squeeze(-1))

    def predict(self, arrays: dict, batch_size: int) -> np.ndarray:
        output = np.empty((len(arrays["lengths"]), len(self.items)), dtype=np.float32)
        for start in range(0, len(output), batch_size):
            indices = np.arange(start, min(start + batch_size, len(output)))
            output[indices] = self.score_batch(arrays, indices).cpu().numpy()
            if (start // batch_size + 1) % 32 == 0:
                print(json.dumps({"stage": "rl_reward_cache_progress", "completed_contexts": int(indices[-1]) + 1,
                                  "total_contexts": len(output)}), flush=True)
        if not np.all(np.isfinite(output)) or np.any((output < 0) | (output > 1)):
            raise ValueError("Invalid catalog reward scores.")
        return output
