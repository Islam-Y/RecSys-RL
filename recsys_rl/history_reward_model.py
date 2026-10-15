from __future__ import annotations

import numpy as np
import torch
from torch import nn


def prior_logits(features: torch.Tensor) -> torch.Tensor:
    values = features[:, [0, 8]].clamp(.0001, .9999)
    return torch.logit(values).mean(dim=1)


class HistoryCompletion(nn.Module):
    def __init__(self, catalog_size: int, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.head = nn.Linear(16, 1)
            self.item_embedding = None
        elif kind == "embedding":
            self.item_embedding = nn.Embedding(catalog_size+3, 16, padding_idx=0)
            nn.init.normal_(self.item_embedding.weight, std=.02)
            with torch.no_grad():
                self.item_embedding.weight[0].zero_()
            self.head = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Dropout(.1),
                                      nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        else:
            raise ValueError(f"Unknown history reward kind: {kind}")
        final = self.head if kind == "linear" else self.head[-1]
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, candidates: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        representation = features
        if self.item_embedding is not None:
            positions = torch.arange(inputs.shape[1], device=inputs.device)[None, :]
            mask = (positions > 0) & (positions < lengths[:, None])
            embedded = self.item_embedding(inputs)
            history = (embedded*mask[:, :, None]).sum(dim=1)/(lengths-1)[:, None]
            last = embedded[torch.arange(len(lengths), device=inputs.device), lengths-1]
            candidate = self.item_embedding(candidates)
            representation = torch.cat((features, history, last, candidate, history*candidate), dim=1)
        return prior_logits(features)+self.head(representation).squeeze(-1)


def batch(arrays: dict, features: np.ndarray, indices: np.ndarray, device: torch.device) -> tuple:
    maximum = int(arrays["lengths"][indices].max())
    return (torch.as_tensor(arrays["input_ids"][indices, :maximum].astype(np.int64), device=device),
            torch.as_tensor(arrays["lengths"][indices].astype(np.int64), device=device),
            torch.as_tensor(arrays["candidate_tokens"][indices].astype(np.int64), device=device),
            torch.as_tensor(features[indices], device=device))


@torch.no_grad()
def predict(model: HistoryCompletion, arrays: dict, features: np.ndarray, device: torch.device,
            batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(features), batch_size):
        indices = np.arange(start, min(start+batch_size, len(features)))
        output.append(torch.sigmoid(model(*batch(arrays, features, indices, device))).cpu().numpy())
    values = np.concatenate(output).astype(np.float32)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid history reward prediction.")
    return values
