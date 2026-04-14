from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ExperimentConfig


def has_cuda() -> bool:
    return torch.cuda.is_available()


def preferred_dtype() -> torch.dtype:
    if not has_cuda():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: ExperimentConfig, checkpoint_path: str | Path | None = None):
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if config.load_in_4bit:
        if not has_cuda():
            raise RuntimeError(
                "load_in_4bit=True requires CUDA. "
                "Use a GPU server to run this config, or switch to a smoke config."
            )
        compute_dtype = preferred_dtype()
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["dtype"] = compute_dtype
    else:
        dtype = preferred_dtype()
        if has_cuda():
            model_kwargs["device_map"] = "auto"
            model_kwargs["dtype"] = dtype
        else:
            model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if checkpoint_path:
        model = PeftModel.from_pretrained(model, str(checkpoint_path))

    return model


def build_lora_config(config: ExperimentConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def count_parameters(model) -> dict[str, float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        size = parameter.numel()
        total += size
        if parameter.requires_grad:
            trainable += size

    ratio = 0.0 if total == 0 else trainable / total * 100
    return {
        "trainable_params": trainable,
        "all_params": total,
        "trainable_ratio_percent": round(ratio, 4),
    }


def get_memory_stats() -> dict[str, float]:
    if not has_cuda():
        return {"peak_gpu_memory_gb": 0.0}
    peak_bytes = torch.cuda.max_memory_allocated()
    return {"peak_gpu_memory_gb": round(peak_bytes / 1024**3, 4)}


def cleanup_memory() -> None:
    gc.collect()
    if has_cuda():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def infer_device(model) -> torch.device:
    return next(model.parameters()).device
