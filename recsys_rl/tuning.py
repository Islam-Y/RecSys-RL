from __future__ import annotations

import copy
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys_rl.data import load_config, load_partition, sha256, write_json
from recsys_rl.evaluation import control_rankings, evaluate_rankings


def source_hashes() -> dict:
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def load_plan(path: Path) -> tuple[dict, dict, Path]:
    plan = json.loads(path.read_text())
    base_path = path.parent / plan["base_config"]
    base = load_config(base_path)
    names = [trial["name"] for trial in plan["trials"]]
    if len(names) < 2 or len(set(names)) != len(names) or "control" not in names:
        raise ValueError("Tuning requires unique trials, including control and a challenger.")
    if any(re.fullmatch(r"[a-z][a-z0-9-]*", name) is None for name in names):
        raise ValueError("Trial names must use lowercase letters, digits and hyphens.")
    for trial in plan["trials"]:
        if set(trial["model"]) - {"dropout"} or set(trial["training"]) - {"learning_rate", "weight_decay"}:
            raise ValueError("This tuning stage varies only dropout, learning rate and weight decay.")
        if trial["name"] == "control" and (trial["model"] or trial["training"]):
            raise ValueError("Control must preserve the original training recipe.")
    seeds = plan["confirmation_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or plan["screening_seed"] not in seeds:
        raise ValueError("Confirmation requires three distinct seeds, including the screening seed.")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Invalid training seed.")
    return plan, base, base_path


def trial_config(base: dict, plan: dict, trial: dict, seed: int) -> dict:
    config = copy.deepcopy(base)
    config["experiment"] = f"{plan['experiment']}-{trial['name']}-seed-{seed}"
    config["model"].update(trial["model"])
    config["training"].update(plan["training"])
    config["training"].update(trial["training"])
    config["training"]["seed"] = seed
    return config


def audit_run(output: Path, config_path: Path, protocol_path: Path, manifest_hash: str,
              expected: dict, validation: dict, catalog: dict, controls: dict) -> dict:
    result = json.loads((output / "summary.json").read_text())
    if result["config"] != expected or result["config_sha256"] != sha256(config_path):
        raise ValueError(f"Run configuration changed: {output}")
    if result["protocol_sha256"] != sha256(protocol_path) or result["preparation_manifest_sha256"] != manifest_hash:
        raise ValueError(f"Run protocol or data changed: {output}")
    if result["training_seed"] != expected["training"]["seed"] or result["data_seed"] != expected["seed"]:
        raise ValueError(f"Data and training seeds differ from the plan: {output}")
    if result["test_evaluated"] or result["reward_model_trained"] or result["rl_trained"]:
        raise ValueError("Unexpected experiment stage flags.")
    if result["model_sha256"] != sha256(output / "baseline.pt"):
        raise ValueError(f"Checkpoint changed: {output}")
    with np.load(output / "validation-top-items.npz", allow_pickle=False) as saved:
        for field in ("user_ids", "timestamps", "target_tokens", "target_offsets"):
            if not np.array_equal(saved[field], validation[field]):
                raise ValueError(f"Saved evaluation population changed: {output}/{field}")
        recalculated = evaluate_rankings(validation, saved["rankings"], catalog, expected["evaluation"]["top_k"])
    if recalculated != result["validation"]["gpt2"]:
        raise ValueError(f"Saved ranking metrics differ: {output}")
    for name, metrics in controls.items():
        if result["validation"][name] != metrics:
            raise ValueError(f"Control differs: {output}/{name}")
    selected = max(result["training_history"], key=lambda row: row["validation"]["metrics"]["ndcg@10"])
    if selected["epoch"] != result["selected_epoch"] or selected["validation"] != recalculated:
        raise ValueError(f"Checkpoint selection differs: {output}")
    return result


def execute_run(config: dict, name: str, seed: int, context: dict) -> dict:
    root = context["output"]
    identifier = f"{name}-seed-{seed}"
    config_path = root / "configs" / f"{identifier}.json"
    output = root / "runs" / name / f"seed-{seed}"
    log_path = root / "logs" / f"{identifier}.log"
    record_path = root / "records" / f"{identifier}.json"
    frozen = context["frozen"]
    if source_hashes() != frozen["source_files_sha256"]:
        raise ValueError("Experiment source changed during tuning.")
    if sha256(context["protocol"]) != frozen["protocol_sha256"] or sha256(context["data"] / "manifest.json") != frozen["preparation_manifest_sha256"]:
        raise ValueError("Protocol or preparation changed during tuning.")
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Saved configuration differs: {config_path}")
    else:
        write_json(config_path, config)
    load_config(config_path)

    def audit() -> dict:
        return audit_run(output, config_path, context["protocol"], frozen["preparation_manifest_sha256"],
                         config, context["validation"], context["catalog"], context["controls"])

    if record_path.exists():
        record = json.loads(record_path.read_text())
        if audit() != record["summary"]:
            raise ValueError(f"Completed run changed: {identifier}")
        for artifact in record["artifacts"].values():
            if sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError(f"Completed artifact changed: {artifact['path']}")
        print(json.dumps({"stage": "tuning_run_reused", "trial": name, "training_seed": seed}), flush=True)
        return record
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete output must be preserved before retrying: {output}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    arguments = ["-u", "-m", "recsys_rl", "baseline", "--data-dir", str(context["data"]),
                 "--config", str(config_path), "--protocol", str(context["protocol"]),
                 "--output-dir", str(output), "--device", context["device"]]
    command = [sys.executable, *arguments]
    external_time = platform.system() == "Darwin" and Path("/usr/bin/time").is_file()
    if external_time:
        command = ["/usr/bin/time", "-l", *command]
    started = time.perf_counter()
    print(json.dumps({"stage": "tuning_run_started", "trial": name, "training_seed": seed}), flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                                   start_new_session=os.name == "posix")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("stage") in {"training_started", "epoch_completed", "early_stopping"}:
                    print(json.dumps({"trial": name, "training_seed": seed, **event}), flush=True)
            return_code = process.wait()
        except BaseException:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
    wall_seconds = time.perf_counter() - started
    if return_code != 0:
        raise ValueError(f"Training exited with {return_code}; see {log_path}")
    result = audit()
    log_text = log_path.read_text()
    measured = re.search(r"^\s*([\d.]+)\s+real\s+[\d.]+\s+user\s+[\d.]+\s+sys\s*$", log_text, re.MULTILINE)
    rss = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", log_text, re.MULTILINE)
    footprint = re.search(r"^\s*(\d+)\s+peak memory footprint\s*$", log_text, re.MULTILINE)
    if external_time and (measured is None or rss is None or footprint is None):
        raise ValueError(f"Missing process accounting: {log_path}")
    record = {
        "trial": name, "training_seed": seed, "summary": result,
        "command": ["python", *arguments],
        "process_measurement": {
            "wall_seconds": wall_seconds,
            "external_real_seconds": float(measured[1]) if measured else None,
            "maximum_resident_set_size_bytes": int(rss[1]) if rss else None,
            "peak_memory_footprint_bytes": int(footprint[1]) if footprint else None,
            "scope": "Child process accounting; not whole-system or complete MPS peak memory."
        },
        "ranking_metrics_recalculated": True, "controls_recalculated": True,
        "artifacts": {filename: {"path": str(output / filename), "sha256": sha256(output / filename),
                                  "bytes": (output / filename).stat().st_size}
                      for filename in ("baseline.pt", "validation-top-items.npz", "training-history.json", "summary.json")}
    }
    record["artifacts"]["log"] = {"path": str(log_path), "sha256": sha256(log_path), "bytes": log_path.stat().st_size}
    write_json(record_path, record)
    print(json.dumps({"stage": "tuning_run_completed", "trial": name, "training_seed": seed,
                      "selected_epoch": result["selected_epoch"], "epochs_completed": result["epochs_completed"],
                      "metrics": result["validation"]["gpt2"]["metrics"], "seconds": wall_seconds}), flush=True)
    return record


def metrics(record: dict) -> dict:
    return record["summary"]["validation"]["gpt2"]["metrics"]


def aggregate(records: list[dict]) -> dict:
    return {name: {"mean": statistics.mean(metrics(record)[name] for record in records),
                   "sample_std": statistics.stdev(metrics(record)[name] for record in records)}
            for name in metrics(records[0])}


def run_tuning(data_dir: Path, output_dir: Path, config_path: Path, protocol_path: Path,
               requested_device: str, resume: bool) -> dict:
    plan, base, base_path = load_plan(config_path)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if base["data"] != manifest["config"]["data"] or base["seed"] != manifest["config"]["seed"]:
        raise ValueError("Base data settings differ from the preparation.")
    reference_path = config_path.parent / plan["reference_result"]
    reference = json.loads(reference_path.read_text())
    if reference["preparation_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Historical comparison used different prepared data.")
    raw_path = config_path.parent / plan["raw_source"]
    if sha256(raw_path) != manifest["source"]["sha256"]:
        raise ValueError("Raw source differs from the frozen preparation.")
    frozen = {
        "config": plan, "tuning_config_sha256": sha256(config_path),
        "base_config_sha256": sha256(base_path), "reference_result_sha256": sha256(reference_path),
        "protocol_sha256": sha256(protocol_path), "preparation_manifest_sha256": sha256(manifest_path),
        "source_files_sha256": source_hashes(), "raw_source_sha256": manifest["source"]["sha256"]
    }
    plan_path = output_dir / "plan.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Tuning output is not empty; use --resume for the same plan: {output_dir}")
        provenance = json.loads(plan_path.read_text())
        if provenance["frozen"] != frozen:
            raise ValueError("Resume plan, source code, data or protocol differs from the frozen run.")
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        provenance = {"frozen": frozen, "source_commit": commit,
                      "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(plan_path, provenance)
    validation = load_partition(data_dir, "validation")
    catalog = json.loads((data_dir / "catalog.json").read_text())
    if sha256(data_dir / "catalog.json") != manifest["catalog_sha256"]:
        raise ValueError("Catalog differs from the frozen preparation.")
    controls = {name: evaluate_rankings(validation, ranking, catalog, base["evaluation"]["top_k"])
                for name, ranking in control_rankings(validation, catalog, max(base["evaluation"]["top_k"])).items()}
    historical = reference["runs"]["small_long"]
    for name, value in controls.items():
        if historical["validation"][name] != value:
            raise ValueError("Controls differ from the historical comparison.")
    context = {"output": output_dir, "data": data_dir, "protocol": protocol_path, "frozen": frozen,
               "device": requested_device, "validation": validation, "catalog": catalog, "controls": controls}
    trials = {trial["name"]: trial for trial in plan["trials"]}
    records = {}
    screening = []
    started = time.perf_counter()
    for trial in plan["trials"]:
        name, seed = trial["name"], plan["screening_seed"]
        record = execute_run(trial_config(base, plan, trial, seed), name, seed, context)
        records[f"{name}/seed-{seed}"] = record
        screening.append(record)
        write_json(output_dir / "progress.json", {"phase": "screening", "completed_runs": list(records)})
    winner = max(screening, key=lambda record: metrics(record)["ndcg@10"])["trial"]
    challenger = max((record for record in screening if record["trial"] != "control"),
                     key=lambda record: metrics(record)["ndcg@10"])["trial"]
    print(json.dumps({"stage": "screening_complete", "winner": winner, "challenger": challenger}), flush=True)
    for seed in plan["confirmation_seeds"]:
        for name in ("control", challenger):
            key = f"{name}/seed-{seed}"
            if key not in records:
                records[key] = execute_run(trial_config(base, plan, trials[name], seed), name, seed, context)
                write_json(output_dir / "progress.json", {"phase": "confirmation", "challenger": challenger,
                                                          "completed_runs": list(records)})
    confirmation = {name: [records[f"{name}/seed-{seed}"] for seed in plan["confirmation_seeds"]]
                    for name in ("control", challenger)}
    paired = []
    for seed in plan["confirmation_seeds"]:
        control_metrics = metrics(records[f"control/seed-{seed}"])
        candidate_metrics = metrics(records[f"{challenger}/seed-{seed}"])
        paired.append({"training_seed": seed,
                       "absolute_differences": {key: candidate_metrics[key] - value for key, value in control_metrics.items()},
                       "relative_percent": {key: 100 * (candidate_metrics[key] / value - 1) if value else None
                                            for key, value in control_metrics.items()}})
    selected_variant = challenger if all(row["absolute_differences"]["ndcg@10"] > 0 for row in paired) else "control"
    selected_record = records[f"{selected_variant}/seed-{plan['screening_seed']}"]
    for split, values in manifest["partitions"].items():
        if sha256(data_dir / f"{split}.npz") != values["sha256"]:
            raise ValueError(f"Prepared partition changed: {split}")
    if sha256(raw_path) != frozen["raw_source_sha256"] or sha256(data_dir / "catalog.json") != manifest["catalog_sha256"]:
        raise ValueError("Raw data or catalog changed.")
    for path, key in ((config_path, "tuning_config_sha256"), (base_path, "base_config_sha256"),
                      (reference_path, "reference_result_sha256"), (protocol_path, "protocol_sha256"),
                      (manifest_path, "preparation_manifest_sha256")):
        if sha256(path) != frozen[key]:
            raise ValueError(f"Frozen experiment input changed: {path}")
    if source_hashes() != frozen["source_files_sha256"]:
        raise ValueError("Experiment code changed before completion.")
    control_screen = records[f"control/seed-{plan['screening_seed']}"]
    old_history = historical["training_history"]
    control_history = control_screen["summary"]["training_history"]
    compared = min(len(old_history), len(control_history))
    historical_differences = [abs(old_history[index]["validation"]["metrics"]["ndcg@10"] -
                                  control_history[index]["validation"]["metrics"]["ndcg@10"])
                              for index in range(compared)]
    result = {
        "experiment": plan["experiment"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance, "data_seed": base["seed"], "screening_seed": plan["screening_seed"],
        "confirmation_seeds": plan["confirmation_seeds"], "runs": records,
        "screening_order": [record["trial"] for record in screening],
        "screening_winner": winner, "screening_challenger": challenger,
        "confirmation": {name: aggregate(values) for name, values in confirmation.items()},
        "paired_differences": paired,
        "mean_paired_differences": {name: statistics.mean(row["absolute_differences"][name] for row in paired)
                                    for name in metrics(selected_record)},
        "adoption_rule": "Adopt the best non-control screening recipe only if its NDCG@10 is strictly higher than control on all three paired training seeds.",
        "selected_variant": selected_variant, "selected_training_seed": plan["screening_seed"],
        "selected_checkpoint": selected_record["artifacts"]["baseline.pt"],
        "historical_control_check": {"epochs_compared": compared,
                                     "maximum_absolute_ndcg_at_10_difference": max(historical_differences)},
        "control_screen_minus_historical_15_epochs": {
            name: value - historical["validation"]["gpt2"]["metrics"][name]
            for name, value in metrics(control_screen).items()},
        "controls": controls, "current_invocation_seconds": time.perf_counter() - started,
        "sum_training_process_wall_seconds": sum(record["process_measurement"]["wall_seconds"] for record in records.values()),
        "raw_data_and_prepared_files_unchanged": True, "test_file_checksum_unchanged": True,
        "source_hashes_checked_before_each_run": True,
        "limits": [
            "Eight single-factor recipes; interactions between hyperparameters were not searched.",
            "Architecture and checkpoints selected on the same validation; confirmation changes only training randomness, not the user/time population.",
            "Three paired training seeds describe run variation; sample standard deviation is not a confidence interval or independent test confirmation.",
            "Same maximum epoch budget and stopping rule; completed epochs and compute can differ.",
            "Conditional evaluation on the same restricted catalog, with substantial UNK histories.",
            "No measured user satisfaction, causal retention uplift, reward model or RL results."
        ],
        "test_evaluated": False, "reward_model_trained": False, "rl_trained": False
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "progress.json", {"phase": "complete", "selected_variant": selected_variant,
                                              "completed_runs": list(records)})
    return result
