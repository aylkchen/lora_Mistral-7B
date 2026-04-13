from __future__ import annotations

import argparse
import json

from lora_lab.runtime import ensure_utf8_runtime

ensure_utf8_runtime()

from lora_lab.config import load_config
from lora_lab.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA + SFT experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint directory for resuming training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_training(config, resume_from_checkpoint=args.resume_from_checkpoint)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

