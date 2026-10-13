from __future__ import annotations

import json
import platform
from pathlib import Path

import duckdb
import numpy as np

from recsys_rl.data import FIRST_ITEM, load_config, load_partition, sha256


def _check_arrays(arrays: dict, catalog_size: int, expected_rows: int, minimum_history: int) -> None:
    inputs, lengths = arrays["input_ids"], arrays["lengths"]
    targets, offsets = arrays["target_tokens"], arrays["target_offsets"]
    if inputs.ndim != 2 or len(inputs) != expected_rows or lengths.shape != (expected_rows,):
        raise ValueError("Prepared input shapes differ from their manifest.")
    if np.any(lengths < minimum_history + 1) or np.any(lengths > inputs.shape[1]):
        raise ValueError("Prepared context lengths are out of bounds.")
    if not np.all(inputs[:, 0] == 1) or np.any(inputs < 0) or np.any(inputs >= catalog_size + FIRST_ITEM):
        raise ValueError("Invalid BOS or input token.")
    padding = np.arange(inputs.shape[1])[None, :] >= lengths[:, None]
    if np.any(inputs[padding] != 0) or np.any(inputs[~padding] == 0):
        raise ValueError("Padding does not match context lengths.")
    if offsets.shape != (expected_rows + 1,) or offsets[0] != 0 or offsets[-1] != len(targets) or np.any(np.diff(offsets) < 1):
        raise ValueError("Target offsets are invalid or a query has no target.")
    if np.any(targets < FIRST_ITEM) or np.any(targets >= catalog_size + FIRST_ITEM):
        raise ValueError("Prepared target is outside the training catalog.")
    keys = (arrays["user_ids"].astype(np.uint64) << 32) | arrays["timestamps"].astype(np.uint64)
    if len(np.unique(keys)) != len(inputs):
        raise ValueError("A prepared partition contains duplicate query IDs.")


def readiness(data_dir: Path, config_path: Path, protocol_path: Path) -> dict:
    config = load_config(config_path)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if sha256(config_path) != manifest["config_sha256"] or sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Configuration or protocol changed after data preparation.")
    if sha256(data_dir / "catalog.json") != manifest["catalog_sha256"]:
        raise ValueError("Training catalog hash differs from the preparation.")
    train = load_partition(data_dir, "train")
    validation = load_partition(data_dir, "validation")
    for split, arrays in (("train", train), ("validation", validation)):
        _check_arrays(arrays, manifest["catalog_size"], manifest["partitions"][split]["queries"], config["data"]["minimum_history"])
    train_end = manifest["cutoffs_seconds"]["train_end"]
    validation_end = manifest["cutoffs_seconds"]["validation_end"]
    if np.any(train["timestamps"] > train_end) or np.any(validation["timestamps"] <= train_end) or np.any(validation["timestamps"] > validation_end):
        raise ValueError("Prepared partitions violate global temporal boundaries.")
    if sha256(data_dir / "test.npz") != manifest["partitions"]["test"]["sha256"]:
        raise ValueError("Held-out test file hash differs; test labels were not loaded for evaluation.")

    # Import framework definitions and inspect device availability; do not instantiate a model.
    import torch
    import transformers
    from transformers import GPT2Config, GPT2LMHeadModel

    if GPT2Config.model_type != "gpt2" or GPT2LMHeadModel.__name__ != "GPT2LMHeadModel":
        raise ValueError("The required GPT-2 implementation is unavailable.")
    mps = bool(torch.backends.mps.is_available())
    cuda = bool(torch.cuda.is_available())
    return {
        "status": "prepared_awaiting_model_start",
        "config_sha256": manifest["config_sha256"], "protocol_sha256": manifest["protocol_sha256"],
        "data": {key: manifest[key] for key in ("source", "cutoffs_seconds", "population_users", "catalog_size", "partitions", "ranking_scope")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "duckdb": duckdb.__version__, "numpy": np.__version__,
            "mps_built": bool(torch.backends.mps.is_built()), "mps_available": mps, "cuda_available": cuda,
            "available_device": "cuda" if cuda else "mps" if mps else "cpu"},
        "checks": {"config_and_protocol_hashes": True, "catalog_and_partition_hashes": True,
            "train_validation_shapes_and_tokens": True, "context_strictly_before_target": True,
            "global_time_boundaries": True, "gpt2_class_imports": True},
        "model_constructed": False, "model_forward_run": False, "training_started": False,
        "test_evaluated": False, "note": "Data and imports are ready. Model construction, forward/backward compatibility, resource usage and quality remain unmeasured until an explicit model run."
    }
