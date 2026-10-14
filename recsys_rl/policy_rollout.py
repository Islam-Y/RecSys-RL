from __future__ import annotations

import json
import numpy as np
import torch

from recsys_rl.data import sha256
from recsys_rl.policy_data import context_rows, new_policy, policy_logits
from recsys_rl.policy_evaluation import append_synthetic_items, log_probabilities, summarize_rollout
from recsys_rl.reward_model import CompletionReward, reward_batch


@torch.no_grad()
def action_rewards(model, arrays: dict, actions: np.ndarray, context: dict) -> np.ndarray:
    inputs = {**arrays, "candidate_tokens": actions}
    output = []
    batch = context["config"]["execution"]["inference_batch_size"]
    for start in range(0, len(actions), batch):
        indices = np.arange(start, min(start + batch, len(actions)))
        output.append(model(*reward_batch(inputs, indices, context["known_users"], context["device"])).cpu().numpy())
    return np.concatenate(output)


def choose_actions(logits: np.ndarray, mode: str, uniforms: np.ndarray) -> np.ndarray:
    if mode == "greedy":
        return np.argmax(logits, axis=1).astype(np.int32) + 3
    if mode == "sampled":
        cumulative = np.cumsum(np.exp(log_probabilities(logits)), axis=1)
        cumulative[:, -1] = 1
        return np.sum(cumulative < uniforms[:, None], axis=1).astype(np.int32) + 3
    raise ValueError(f"Unknown rollout mode: {mode}")


def generate_rollouts(context: dict, runs: dict) -> dict:
    root = context["output"] / "rollouts"
    root.mkdir(parents=True, exist_ok=True)
    with np.load(context["output"] / "cache" / "rollout-inputs.npz", allow_pickle=False) as source:
        indices, uniforms = source["indices"].copy(), source["uniforms"].copy()
    original = context_rows(context["validation"], indices)
    checkpoint = torch.load(context["reward_dir"] / "reward.pt", map_location="cpu", weights_only=True)
    model = CompletionReward(context["catalog"], context["policy_config"]["data"]["history_length"], context["known_users"], context["reward_report"]["config"]["model"]).to(context["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    policies = {"baseline": None, **{variant["name"]: runs[f"{variant['name']}/seed-{context['config']['training_seed']}"]
                                       for variant in context["config"]["variants"]}}
    personal = context["personal"]["validation"][indices]
    settings, calibration = context["config"], context["calibration"]
    results = {}
    for name, run in policies.items():
        policy = new_policy(context)
        checkpoint_path = context["policy_checkpoint"]
        if run is not None:
            checkpoint_path = context["output"] / "runs" / name / f"seed-{run['training_seed']}" / "policy.pt"
            policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
        checkpoint_sha = sha256(checkpoint_path)
        results[name] = {}
        for mode in settings["rollout"]["modes"]:
            path = root / f"{name}-{mode}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    payload = dict(saved)
                if not np.array_equal(payload["indices"], indices) or not np.array_equal(payload["uniforms"], uniforms):
                    raise ValueError("Saved rollout uses different population or common random numbers.")
                if str(payload["checkpoint_sha256"]) != checkpoint_sha:
                    raise ValueError("Saved rollout uses a different policy checkpoint.")
            else:
                current = {key: value.copy() for key, value in original.items()}
                actions, raw_rewards, hybrid_rewards, time_rewards = [], [], [], []
                for step in range(settings["rollout"]["steps"]):
                    logits = policy_logits(policy, current, context["device"], settings["execution"]["inference_batch_size"])
                    chosen = choose_actions(logits, mode, uniforms[:, step])
                    raw = action_rewards(model, current, chosen, context)
                    calibrated = np.clip(calibration["slope"] * raw.astype(np.float64) + calibration["intercept"], 0, 1)
                    hybrid = personal + calibration["personal_mix_alpha"] * (calibrated - personal)
                    actions.append(chosen)
                    raw_rewards.append(raw)
                    hybrid_rewards.append(hybrid)
                    time_rewards.append(hybrid * context["duration_scale"][chosen - 3])
                    current["input_ids"], current["lengths"] = append_synthetic_items(current["input_ids"], current["lengths"], chosen)
                payload = {"indices": indices, "uniforms": uniforms, "checkpoint_sha256": np.asarray(checkpoint_sha),
                           "user_ids": original["user_ids"], "timestamps": original["timestamps"],
                           "actions": np.stack(actions, axis=1), "raw_reward": np.stack(raw_rewards, axis=1),
                           "hybrid_reward": np.stack(hybrid_rewards, axis=1), "time_reward": np.stack(time_rewards, axis=1)}
                np.savez_compressed(path, **payload)
            current = {key: value.copy() for key, value in original.items()}
            for field in ("user_ids", "timestamps"):
                if not np.array_equal(payload[field], original[field]):
                    raise ValueError("Saved rollout population differs.")
            for step in range(settings["rollout"]["steps"]):
                chosen = payload["actions"][:, step]
                if np.any((chosen < 3) | (chosen >= len(context["catalog"]["items"]) + 3)):
                    raise ValueError("Rollout contains an invalid item.")
                logits = policy_logits(policy, current, context["device"], settings["execution"]["inference_batch_size"])
                if not np.array_equal(chosen, choose_actions(logits, mode, uniforms[:, step])):
                    raise ValueError("Saved rollout actions do not reproduce from the policy.")
                reproduced = action_rewards(model, current, chosen, context)
                if not np.allclose(reproduced, payload["raw_reward"][:, step], atol=1e-6, rtol=0):
                    raise ValueError("Saved synthetic rollout reward does not reproduce.")
                calibrated = np.clip(calibration["slope"] * reproduced.astype(np.float64) + calibration["intercept"], 0, 1)
                hybrid = personal + calibration["personal_mix_alpha"] * (calibrated - personal)
                if not np.allclose(hybrid, payload["hybrid_reward"][:, step], atol=1e-6, rtol=0) or not np.allclose(hybrid * context["duration_scale"][chosen - 3], payload["time_reward"][:, step], atol=1e-6, rtol=0):
                    raise ValueError("Saved rollout reward transformation differs.")
                current["input_ids"], current["lengths"] = append_synthetic_items(current["input_ids"], current["lengths"], chosen)
            results[name][mode] = {"summary": summarize_rollout(original, payload, context["catalog"], context["counts"], settings["evaluation"]["low_support_threshold"]),
                                   "artifact": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
                                   "saved_reward_replayed": True, "saved_policy_actions_replayed": True}
            print(json.dumps({"stage": "rl_rollout_completed", "variant": name, "mode": mode}), flush=True)
        del policy
        if context["device"].type == "mps":
            torch.mps.empty_cache()
    return results
