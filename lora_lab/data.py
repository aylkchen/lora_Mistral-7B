from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_splits(
    dataset_path: str | Path,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetDict:
    dataset_path = Path(dataset_path)

    if dataset_path.is_dir():
        split_candidates = {
            "train": dataset_path / "train.jsonl",
            "validation": dataset_path / "val.jsonl",
            "test": dataset_path / "test.jsonl",
        }
        if all(path.exists() for path in split_candidates.values()):
            return load_dataset(
                "json",
                data_files={key: str(value) for key, value in split_candidates.items()},
            )

    data_files = {"train": str(dataset_path)}
    dataset = load_dataset("json", data_files=data_files)["train"]
    test_split = dataset.train_test_split(test_size=test_ratio, seed=seed)
    val_share = val_ratio / (1 - test_ratio)
    train_val = test_split["train"].train_test_split(test_size=val_share, seed=seed)
    return DatasetDict(
        train=train_val["train"],
        validation=train_val["test"],
        test=test_split["test"],
    )


def apply_sample_limits(
    dataset: DatasetDict,
    train_limit: int | None,
    eval_limit: int | None,
    test_limit: int | None,
) -> DatasetDict:
    limited = DatasetDict()
    if "train" in dataset:
        limited["train"] = dataset["train"].select(
            range(min(len(dataset["train"]), train_limit or len(dataset["train"])))
        )
    if "validation" in dataset:
        limited["validation"] = dataset["validation"].select(
            range(
                min(
                    len(dataset["validation"]),
                    eval_limit or len(dataset["validation"]),
                )
            )
        )
    if "test" in dataset:
        limited["test"] = dataset["test"].select(
            range(min(len(dataset["test"]), test_limit or len(dataset["test"])))
        )
    return limited


def render_chat_text(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    assistant_response: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    parts: list[str] = []
    if system_prompt:
        parts.append(f"<SYSTEM>\n{system_prompt}")
    parts.append(f"<USER>\n{user_prompt}")
    if assistant_response is not None:
        parts.append(f"<ASSISTANT>\n{assistant_response}")
    elif add_generation_prompt:
        parts.append("<ASSISTANT>\n")
    return "\n\n".join(parts)


def add_text_column(dataset: DatasetDict, tokenizer: Any) -> DatasetDict:
    def _formatter(example: dict[str, Any]) -> dict[str, Any]:
        text = render_chat_text(
            tokenizer=tokenizer,
            system_prompt=example.get("system", ""),
            user_prompt=example["user"],
            assistant_response=example.get("assistant"),
            add_generation_prompt=False,
        )
        return {"text": text}

    mapped = DatasetDict()
    for split_name, split_dataset in dataset.items():
        mapped[split_name] = split_dataset.map(_formatter)
    return mapped

