from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential next-item recommendation experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare point-in-time Yambda sequences and a train-only catalog.")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed/mvp"))
    prepare.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    prepare.add_argument("--protocol", type=Path, default=Path("experiments/baseline-protocol.md"))
    baseline = commands.add_parser("baseline", help="Train GPT-2 and evaluate validation; leave test unopened.")
    baseline.add_argument("--data-dir", type=Path, default=Path("data/processed/mvp"))
    baseline.add_argument("--output-dir", type=Path, default=Path("outputs/baseline-v1"))
    baseline.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    baseline.add_argument("--protocol", type=Path, help="Protocol for this training variant; defaults to the preparation protocol.")
    baseline.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    tune = commands.add_parser("tune", help="Run bounded small-policy tuning and paired training-seed confirmation.")
    tune.add_argument("--data-dir", type=Path, default=Path("data/processed/mvp"))
    tune.add_argument("--output-dir", type=Path, default=Path("outputs/small-tuning-v1"))
    tune.add_argument("--config", type=Path, default=Path("configs/small-tuning.json"))
    tune.add_argument("--protocol", type=Path, default=Path("experiments/small-tuning-protocol.md"))
    tune.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    tune.add_argument("--resume", action="store_true", help="Reuse completed, audited runs of the same frozen plan.")
    reward_prepare = commands.add_parser("reward-prepare", help="Attach observed completion labels and split policy train chronologically.")
    reward_prepare.add_argument("input", type=Path)
    reward_prepare.add_argument("--data-dir", type=Path, default=Path("data/processed/mvp"))
    reward_prepare.add_argument("--output-dir", type=Path, default=Path("data/processed/reward-v1"))
    reward_prepare.add_argument("--config", type=Path, default=Path("configs/reward.json"))
    reward_prepare.add_argument("--protocol", type=Path, default=Path("experiments/reward-protocol.md"))
    reward = commands.add_parser("reward", help="Train a completion reward model; evaluate policy validation after selection.")
    reward.add_argument("--data-dir", type=Path, default=Path("data/processed/reward-v1"))
    reward.add_argument("--output-dir", type=Path, default=Path("outputs/reward-v1"))
    reward.add_argument("--config", type=Path, default=Path("configs/reward.json"))
    reward.add_argument("--protocol", type=Path, default=Path("experiments/reward-protocol.md"))
    reward.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    reward_audit = commands.add_parser("reward-audit", help="Recalculate saved reward predictions and verify frozen artifacts.")
    reward_audit.add_argument("--data-dir", type=Path, default=Path("data/processed/reward-v1"))
    reward_audit.add_argument("--output-dir", type=Path, default=Path("outputs/reward-v1"))
    reward_signal = commands.add_parser("reward-signal", help="Audit candidate discrimination and calibrate a frozen reward model.")
    reward_signal.add_argument("--data-dir", type=Path, default=Path("data/processed/reward-v1"))
    reward_signal.add_argument("--output-dir", type=Path, default=Path("outputs/reward-signal-v1"))
    reward_signal.add_argument("--config", type=Path, default=Path("configs/reward-signal.json"))
    reward_signal.add_argument("--protocol", type=Path, default=Path("experiments/reward-signal-protocol.md"))
    reward_signal.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    rl = commands.add_parser("rl", help="Run the frozen six-variant contextual-bandit pilot and five-item rollouts.")
    rl.add_argument("--data-dir", type=Path, default=Path("data/processed/mvp"))
    rl.add_argument("--reward-data-dir", type=Path, default=Path("data/processed/reward-v1"))
    rl.add_argument("--output-dir", type=Path, default=Path("outputs/policy-rl-pilot-v1"))
    rl.add_argument("--config", type=Path, default=Path("configs/rl-pilot.json"))
    rl.add_argument("--protocol", type=Path, default=Path("experiments/rl-pilot-protocol.md"))
    rl.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    rl.add_argument("--resume", action="store_true", help="Reuse completed artifacts of the same frozen plan.")
    rl.add_argument("--prepare-only", action="store_true", help="Cache and validate the frozen models without RL training.")
    rl_audit = commands.add_parser("rl-audit", help="Replay saved policy forwards, controls, selection, and rollouts.")
    rl_audit.add_argument("--output-dir", type=Path, default=Path("outputs/policy-rl-pilot-v1"))
    rl_audit.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    status = commands.add_parser("status", help="Inspect data and dependency readiness without creating or running a model.")
    status.add_argument("--data-dir", type=Path, default=Path("data/processed/mvp"))
    status.add_argument("--config", type=Path, default=Path("configs/mvp.json"))
    status.add_argument("--protocol", type=Path, default=Path("experiments/baseline-protocol.md"))
    status.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from recsys_rl.data import prepare
            result = prepare(args.input, args.output_dir, args.config, args.protocol)
            print(json.dumps({"stage": "preparation_complete", "catalog_size": result["catalog_size"], "partitions": result["partitions"]}, ensure_ascii=False))
        elif args.command == "baseline":
            from recsys_rl.train import train
            result = train(args.data_dir, args.output_dir, args.config, args.device, args.protocol)
            print(json.dumps({"stage": "baseline_complete", "selected_epoch": result["selected_epoch"], "validation": result["validation"], "test_evaluated": False}, ensure_ascii=False))
        elif args.command == "tune":
            from recsys_rl.tuning import run_tuning
            result = run_tuning(args.data_dir, args.output_dir, args.config, args.protocol, args.device, args.resume)
            print(json.dumps({"stage": "tuning_complete", "selected_variant": result["selected_variant"], "screening_winner": result["screening_winner"], "test_evaluated": False}))
        elif args.command == "reward-prepare":
            from recsys_rl.reward_data import prepare_reward
            result = prepare_reward(args.input, args.data_dir, args.output_dir, args.config, args.protocol)
            print(json.dumps({"stage": "reward_preparation_complete", "cutoffs_seconds": result["cutoffs_seconds"],
                              "partitions": result["partitions"], "test_evaluated": False}, ensure_ascii=False))
        elif args.command == "reward":
            from recsys_rl.reward import train_reward
            result = train_reward(args.data_dir, args.output_dir, args.config, args.protocol, args.device)
            print(json.dumps({"stage": "reward_training_complete", "selected_epoch": result["selected_epoch"],
                              "validation": {name: value["metrics"] for name, value in result["validation"].items()},
                              "audit": result["audit"], "test_evaluated": False}, ensure_ascii=False))
        elif args.command == "reward-audit":
            from recsys_rl.reward import audit_reward_run
            print(json.dumps(audit_reward_run(args.data_dir, args.output_dir), ensure_ascii=False))
        elif args.command == "reward-signal":
            from recsys_rl.reward_signal import run_reward_signal
            result = run_reward_signal(args.data_dir, args.output_dir, args.config, args.protocol, args.device)
            print(json.dumps({"stage": "reward_signal_complete", "calibration": result["calibration"],
                              "audit": result["audit"], "rl_trained": False, "test_evaluated": False}))
        elif args.command == "rl":
            from recsys_rl.policy_optimization import run_policy_pilot
            result = run_policy_pilot(args.data_dir, args.reward_data_dir, args.output_dir, args.config,
                                      args.protocol, args.device, args.resume, args.prepare_only)
            print(json.dumps({"stage": result.get("stage", "rl_pilot_complete"),
                              "selected_variant": result.get("selected_variant"), "runs": len(result.get("runs", {})),
                              "rl_trained": result["rl_trained"], "test_evaluated": False}))
        elif args.command == "rl-audit":
            from recsys_rl.policy_optimization import audit_policy_pilot
            print(json.dumps(audit_policy_pilot(args.output_dir, args.device)))
        else:
            from recsys_rl.data import write_json
            from recsys_rl.status import readiness
            result = readiness(args.data_dir, args.config, args.protocol)
            if args.output is not None:
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")
