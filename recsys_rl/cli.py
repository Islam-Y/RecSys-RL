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
        else:
            from recsys_rl.data import write_json
            from recsys_rl.status import readiness
            result = readiness(args.data_dir, args.config, args.protocol)
            if args.output is not None:
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")
