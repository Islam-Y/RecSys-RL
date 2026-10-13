from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available in this runtime.")
    return torch.device(requested)


class NextItemPolicy(nn.Module):
    def __init__(self, catalog_size: int, history_length: int, settings: dict):
        super().__init__()
        config = GPT2Config(vocab_size=catalog_size + 3, n_positions=history_length + 1,
            n_embd=settings["embedding_size"], n_layer=settings["layers"], n_head=settings["heads"],
            resid_pdrop=settings["dropout"], embd_pdrop=settings["dropout"], attn_pdrop=settings["dropout"],
            bos_token_id=1, eos_token_id=None, pad_token_id=0, use_cache=False,
            attn_implementation="eager")
        self.backbone = GPT2LMHeadModel(config)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        hidden = self.backbone.transformer(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(lengths), device=input_ids.device), lengths - 1]
        return self.backbone.lm_head(last)[:, 3:]
