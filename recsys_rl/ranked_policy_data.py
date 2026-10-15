from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostRanker

from recsys_rl.data import load_partition, sha256, write_json
from recsys_rl.history_reward_data import connection, history_features, load_npz
from recsys_rl.model import select_device
from recsys_rl.policy_data import new_policy, policy_logits
from recsys_rl.ranking_model import calibrated
from recsys_rl.recency_data import read_json
from recsys_rl.recency_model import copy_distribution
from recsys_rl.reward import reward_source_hashes
from recsys_rl.reward_data import load_reward_partition
from recsys_rl.reward_model import duration_features


def verify(context: dict) -> None:
    if reward_source_hashes() != context["plan"]["frozen"]["source_files"]: raise ValueError("Ranked-policy source changed.")
    for path, expected in context["plan"]["frozen"]["inputs"].items():
        if sha256(Path(path)) != expected: raise ValueError(f"Ranked-policy input changed: {path}")


def load_context(config_path: Path, protocol: Path, output: Path, device: str, resume: bool) -> dict:
    config_path, protocol, output = [p.resolve() for p in (config_path, protocol, output)]
    config = read_json(config_path); resolve = lambda key: (config_path.parent / config[key]).resolve()
    ranking = read_json(resolve("ranking_report")); execution = read_json(resolve("ranking_execution"))
    if not ranking["rl_allowed"] or sha256(resolve("ranking_report")) != execution["result_sha256"]:
        raise ValueError("Confirmed ranking signal is unavailable.")
    frozen = ranking["provenance"]["frozen"]
    for name, expected in frozen["source_files"].items():
        if sha256(Path(__file__).parent / name) != expected: raise ValueError(f"Historical ranking source changed: {name}")
    inputs = dict(frozen["inputs"])
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected: raise ValueError("Historical ranking input changed.")
    run = ranking["runs"][f"{ranking['selection']['selected_variant']}/42"]
    if sha256(Path(run["checkpoint"])) != run["artifacts"][Path(run["checkpoint"]).name]: raise ValueError("Selected ranker changed.")
    ranking_data = resolve("ranking_data_dir"); final_data = resolve("final_data_dir")
    for name in ("priors.npz", "populations.npz"):
        if sha256(ranking_data / name) != ranking["preparation"]["artifacts"][name]: raise ValueError("Ranking preparation changed.")
    final_manifest = read_json(final_data / "manifest.json")
    for name in ("final.npz", "final-users.npz"):
        if sha256(final_data / name) != final_manifest["artifacts"][name]: raise ValueError("Sealed final reserve changed.")
    for p in (config_path, protocol, resolve("ranking_report"), resolve("ranking_execution"), resolve("recency_execution"),
              Path(run["checkpoint"]), ranking_data / "manifest.json", ranking_data / "priors.npz", ranking_data / "populations.npz",
              final_data / "manifest.json", final_data / "final.npz", final_data / "final-users.npz"):
        inputs[str(p)] = sha256(p)
    own = {"inputs": inputs, "source_files": reward_source_hashes()}
    paths = {"config": str(config_path), "protocol": str(protocol), "output": str(output)}
    if output.exists() and any(output.iterdir()):
        if not resume: raise FileExistsError("Ranked-policy output exists; use --resume or --audit.")
        plan = read_json(output / "plan.json")
        if plan["frozen"] != own or plan["paths"] != paths: raise ValueError("Ranked-policy plan changed.")
    else:
        plan = {"config": config, "frozen": own, "paths": paths,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "plan.json", plan)
    data_dir = next(Path(p).parent for p in inputs if p.endswith("data/processed/mvp/manifest.json"))
    reward_data = next(Path(p).parent for p in inputs if p.endswith("data/processed/reward-v1/manifest.json"))
    policy_checkpoint = next(Path(p) for p in inputs if p.endswith("small-tuning-v1/runs/lr-low/seed-42/baseline.pt"))
    pilot_cache = next(Path(p) for p in inputs if p.endswith("policy-rl-pilot-v1/cache/train.npz"))
    original = load_partition(data_dir, "train"); indices = load_npz(pilot_cache)["original_indices"]
    context = {"config": config, "plan": plan, "output": output, "data_dir": data_dir, "final_data_dir": final_data,
               "catalog": read_json(data_dir / "catalog.json"), "policy_config": read_json(config_path.parent / "baseline-selected.json"),
               "policy_checkpoint": policy_checkpoint, "device": select_device(device), "ranking": ranking, "ranker_run": run,
               "calibration": ranking["calibration"], "priors": load_npz(ranking_data / "priors.npz"),
               "source": next(Path(p) for p in inputs if p.endswith("yambda-50m-multi_event.parquet")),
               "gate": read_json(resolve("recency_execution"))["selected_global_gate"]["sigmoid_bias"],
               "tau": read_json(resolve("recency_execution"))["selected_global_gate"]["variant"]["tau"],
               "train": {k: original[k][indices] for k in ("input_ids", "lengths", "user_ids", "timestamps", "history_max_timestamps")},
               "validation": load_partition(data_dir, "validation"), "observed_validation": load_reward_partition(reward_data, "validation")}
    populations = load_npz(ranking_data / "populations.npz"); reserve = load_npz(final_data / "final-users.npz")
    if np.intersect1d(populations["training_users"], reserve["users"]).size or np.intersect1d(context["train"]["user_ids"], reserve["users"]).size:
        raise ValueError("Final users occur in policy/reward training.")
    torch.set_num_threads(config["execution"]["threads"]); verify(context)
    return context


def mixture(context: dict, arrays: dict, logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64)-logits.max(axis=1, keepdims=True)
    base = np.exp(shifted); base /= base.sum(axis=1, keepdims=True)
    copy = copy_distribution(arrays, logits.shape[1], context["tau"])
    g = np.float32(context["gate"])*(copy.sum(axis=1) > 0).astype(np.float32)
    return (1-g[:, None])*base.astype(np.float32)+g[:, None]*copy


def catalog_reward(context: dict, queries: dict) -> np.ndarray:
    db = connection(context)
    try:
        db.register("requested_users", {"user_id": np.unique(queries["user_ids"])})
        events = db.execute("""SELECT uid AS user_id,timestamp,item_id,
            avg(CASE WHEN played_ratio_pct IS NOT NULL THEN least(played_ratio_pct,100)/100.0 END) AS response
            FROM raw_events JOIN requested_users ON uid=user_id WHERE event_type='listen' AND timestamp<?
            GROUP BY uid,timestamp,item_id ORDER BY uid,timestamp,item_id""", [int(queries["timestamps"].max())]).fetchnumpy()
    finally: db.close()
    response = np.ma.filled(events["response"], np.nan).astype(np.float64)
    items = np.asarray([i["item_id"] for i in context["catalog"]["items"]], dtype=np.uint32); n = len(items)
    ordering = np.argsort(items); at = np.searchsorted(items[ordering], events["item_id"])
    known = (at < n) & (items[ordering][np.minimum(at, n-1)] == events["item_id"])
    tokens = np.full(len(at), 2, dtype=np.int32); tokens[known] = ordering[at[known]]+3
    users, starts, counts = np.unique(events["user_id"], return_index=True, return_counts=True)
    boundaries = {int(u): (int(s), int(s+c)) for u, s, c in zip(users, starts, counts)}
    prior = context["priors"]; y = prior["rewards"].astype(np.float64); mean = y.mean()
    count = np.bincount(prior["candidate_tokens"]-3, minlength=n)
    totals = np.bincount(prior["candidate_tokens"]-3, weights=y, minlength=n)
    item_prior = (totals+20*mean)/(count+20); durations = duration_features(context["catalog"])[3:]
    model = CatBoostRanker(); model.load_model(context["ranker_run"]["checkpoint"])
    output = np.empty((len(queries["lengths"]), n), dtype=np.float32); batch_size = context["config"]["execution"]["reward_batch_size"]
    features = np.empty((batch_size, n, 16), dtype=np.float32)
    for offset in range(0, len(output), batch_size):
        stop_batch = min(offset+batch_size, len(output))
        for local, i in enumerate(range(offset, stop_batch)):
            start, end = boundaries[int(queries["user_ids"][i])]
            stop = start+int(np.searchsorted(events["timestamp"][start:end], queries["timestamps"][i], side="left"))
            lower = max(start, stop-64); h, values = tokens[lower:stop], response[lower:stop]
            valid = np.isfinite(values); observed = values[valid]
            user = (observed.sum()+5*mean)/(len(observed)+5)
            def short(size):
                part = values[-size:]; part = part[np.isfinite(part)]
                return float(part.mean()) if part.size else float(user)
            common = [user, short(4), short(16), float(observed.std()) if observed.size else 0,
                      float((observed >= .9).mean()) if observed.size else user, float((observed == 0).mean()) if observed.size else 0,
                      len(observed)/64, float(np.mean(queries["input_ids"][i, 1:int(queries["lengths"][i])] == 2))]
            f = features[local]; f[:, :8] = common; f[:, 8] = item_prior
            catalog_mask = h >= 3; eligible = catalog_mask & valid
            frequencies = np.bincount(h[catalog_mask]-3, minlength=n)
            observed_counts = np.bincount(h[eligible]-3, minlength=n)
            observed_sums = np.bincount(h[eligible]-3, weights=values[eligible], minlength=n)
            f[:, 9] = (observed_sums+5*user)/(observed_counts+5); f[:, 10] = frequencies/64
            f[:, 11] = 1; f[:, 12] = user
            for position, token in enumerate(h):
                if token >= 3:
                    f[token-3, 11] = (len(h)-1-position)/64
                    if valid[position]: f[token-3, 12] = values[position]
            f[:, 13] = durations[:, 0]/3; f[:, 14] = durations[:, 1]; f[:, 15] = frequencies > 0
        flat = features[:stop_batch-offset].reshape(-1, 16)
        scores = model.predict(flat, thread_count=context["config"]["execution"]["threads"])
        output[offset:stop_batch] = calibrated(scores, flat, context["calibration"]).reshape(-1, n).astype(np.float32)
        if (offset//batch_size+1) % 16 == 0:
            print(f'{{"stage":"ranked_reward_cache","queries":{stop_batch},"total":{len(output)}}}', flush=True)
    if not np.isfinite(output).all() or np.any((output < 0) | (output > 1)): raise ValueError("Invalid calibrated catalog reward.")
    return output


def prepare_cache(context: dict) -> dict:
    root = context["output"] / "cache"
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        for name, expected in manifest["artifacts"].items():
            if sha256(root / name) != expected: raise ValueError("Ranked-policy cache changed.")
        return {s: load_npz(root / f"{s}.npz") for s in ("train", "validation")}
    if root.exists() and any(root.iterdir()): raise FileExistsError("Preserve incomplete ranked-policy cache.")
    root.mkdir(exist_ok=True); caches = {}; started = datetime.now(timezone.utc).isoformat()
    policy = new_policy(context)
    for split in ("train", "validation"):
        reward = catalog_reward(context, context[split])
        logits = policy_logits(policy, context[split], context["device"], context["config"]["execution"]["inference_batch_size"])
        caches[split] = {"reward": reward, "reference_logits": logits}
        np.savez_compressed(root / f"{split}.npz", **caches[split])
    observed = context["observed_validation"]
    features = history_features({**context, "train": context["priors"], "config": {"history": {"events": 64, "user_smoothing": 5, "candidate_smoothing": 5, "item_smoothing": 20}}}, observed, False)["features"]
    model = CatBoostRanker(); model.load_model(context["ranker_run"]["checkpoint"])
    direct = calibrated(model.predict(features, thread_count=context["config"]["execution"]["threads"]), features, context["calibration"])
    difference = float(np.max(np.abs(direct-caches["validation"]["reward"][observed["query_indices"], observed["candidate_tokens"]-3])))
    if difference > context["config"]["evaluation"]["forward_tolerance"]: raise ValueError("Catalog feature/reward forward differs from direct observed forward.")
    write_json(root / "manifest.json", {"started_at_utc": started, "observed_pairs_checked": len(direct), "maximum_direct_reward_difference": difference,
                                       "artifacts": {p.name: sha256(p) for p in root.glob("*.npz")}})
    verify(context)
    return caches
