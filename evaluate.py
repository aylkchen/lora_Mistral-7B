from __future__ import annotations

import argparse
import json

from lora_lab.config import load_config
from lora_lab.evaluation import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Adapter directory saved by train.py.",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Skip base-model comparison and only evaluate the adapter checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_evaluation(
        config=config,
        checkpoint_path=args.checkpoint,
        compare_base=not args.no_base,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
