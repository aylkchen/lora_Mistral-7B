from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .runtime import project_root


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


@dataclass
class LoraSection:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class GenerationSection:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class EvaluationSection:
    mode: str = "faq"
    prompt_path: Path | None = None
    style_cues_path: Path | None = None
    manual_score_path: Path | None = None
    sample_size: int | None = None


@dataclass
class ExperimentConfig:
    base_model: str
    task_name: str
    dataset_path: Path
    output_dir: Path
    load_in_4bit: bool = True
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    seed: int = 42
    logging_steps: int = 20
    save_steps: int = 200
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    report_to: str = "none"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_test_samples: int | None = None
    lora: LoraSection = field(default_factory=LoraSection)
    generation: GenerationSection = field(default_factory=GenerationSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        payload["evaluation"]["prompt_path"] = (
            str(self.evaluation.prompt_path) if self.evaluation.prompt_path else None
        )
        payload["evaluation"]["style_cues_path"] = (
            str(self.evaluation.style_cues_path)
            if self.evaluation.style_cues_path
            else None
        )
        payload["evaluation"]["manual_score_path"] = (
            str(self.evaluation.manual_score_path)
            if self.evaluation.manual_score_path
            else None
        )
        return payload


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    lora = LoraSection(
        r=raw.get("lora", {}).get("r", 16),
        alpha=raw.get("lora", {}).get("alpha", 32),
        dropout=raw.get("lora", {}).get("dropout", 0.05),
        target_modules=raw.get("lora", {}).get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    generation = GenerationSection(**raw.get("generation", {}))
    evaluation_raw = raw.get("evaluation", {})
    evaluation = EvaluationSection(
        mode=evaluation_raw.get("mode", "faq"),
        prompt_path=_resolve_path(evaluation_raw.get("prompt_path")),
        style_cues_path=_resolve_path(evaluation_raw.get("style_cues_path")),
        manual_score_path=_resolve_path(evaluation_raw.get("manual_score_path")),
        sample_size=evaluation_raw.get("sample_size"),
    )

    return ExperimentConfig(
        base_model=raw["base_model"],
        task_name=raw["task_name"],
        dataset_path=_resolve_path(raw["dataset_path"]) or Path(raw["dataset_path"]),
        output_dir=_resolve_path(raw["output_dir"]) or Path(raw["output_dir"]),
        load_in_4bit=raw.get("load_in_4bit", True),
        batch_size=raw.get("batch_size", 4),
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 4),
        epochs=raw.get("epochs", 3),
        learning_rate=raw.get("learning_rate", 2e-4),
        max_seq_length=raw.get("max_seq_length", 512),
        seed=raw.get("seed", 42),
        logging_steps=raw.get("logging_steps", 20),
        save_steps=raw.get("save_steps", 200),
        save_total_limit=raw.get("save_total_limit", 2),
        gradient_checkpointing=raw.get("gradient_checkpointing", True),
        report_to=raw.get("report_to", "none"),
        max_train_samples=raw.get("max_train_samples"),
        max_eval_samples=raw.get("max_eval_samples"),
        max_test_samples=raw.get("max_test_samples"),
        lora=lora,
        generation=generation,
        evaluation=evaluation,
    )

