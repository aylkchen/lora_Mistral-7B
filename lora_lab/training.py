from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import add_text_column, apply_sample_limits, load_dataset_splits
from .modeling import (
    build_lora_config,
    cleanup_memory,
    count_parameters,
    get_memory_stats,
    has_cuda,
    load_model,
    load_tokenizer,
    preferred_dtype,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if has_cuda():
        torch.cuda.manual_seed_all(seed)


def run_training(config: ExperimentConfig, resume_from_checkpoint: str | None = None) -> dict:
    from trl import SFTConfig, SFTTrainer

    _set_seed(config.seed)
    cleanup_memory()

    tokenizer = load_tokenizer(config.base_model)
    dataset = load_dataset_splits(config.dataset_path, seed=config.seed)
    dataset = apply_sample_limits(
        dataset,
        train_limit=config.max_train_samples,
        eval_limit=config.max_eval_samples,
        test_limit=config.max_test_samples,
    )
    dataset = add_text_column(dataset, tokenizer)

    model = load_model(config)
    peft_config = build_lora_config(config)

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        seed=config.seed,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        remove_unused_columns=False,
        packing=False,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="epoch" if "validation" in dataset else "no",
        do_train=True,
        do_eval="validation" in dataset,
        fp16=has_cuda() and preferred_dtype() == torch.float16,
        bf16=has_cuda() and preferred_dtype() == torch.bfloat16,
        use_cpu=not has_cuda(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    param_stats = count_parameters(trainer.model)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    summary = {
        "task_name": config.task_name,
        "dataset_sizes": {
            split_name: len(split_dataset) for split_name, split_dataset in dataset.items()
        },
        "train_metrics": train_result.metrics,
        "parameter_stats": param_stats,
        "memory_stats": get_memory_stats(),
        "config": config.to_dict(),
    }

    if "validation" in dataset:
        summary["validation_metrics"] = trainer.evaluate()

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cleanup_memory()
    return summary
