from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .data import load_dataset_splits, read_jsonl, render_chat_text, write_jsonl
from .modeling import cleanup_memory, infer_device, load_model, load_tokenizer


def _load_prompts(config: ExperimentConfig) -> list[dict[str, Any]]:
    if config.evaluation.prompt_path and config.evaluation.prompt_path.exists():
        prompts = read_jsonl(config.evaluation.prompt_path)
    else:
        dataset = load_dataset_splits(config.dataset_path, seed=config.seed)
        prompts = [dict(row) for row in dataset["test"]]
    if config.evaluation.sample_size:
        prompts = prompts[: config.evaluation.sample_size]
    return prompts


def _generate_response(model, tokenizer, record: dict[str, Any], config: ExperimentConfig) -> str:
    prompt = render_chat_text(
        tokenizer=tokenizer,
        system_prompt=record.get("system", ""),
        user_prompt=record["user"],
        assistant_response=None,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    device = infer_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            repetition_penalty=config.generation.repetition_penalty,
            do_sample=config.generation.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _collect_predictions(
    config: ExperimentConfig,
    prompts: list[dict[str, Any]],
    checkpoint_path: str | Path | None,
) -> list[str]:
    model = load_model(config, checkpoint_path=checkpoint_path)
    tokenizer = load_tokenizer(config.base_model)
    predictions = [_generate_response(model, tokenizer, prompt, config) for prompt in prompts]
    del model
    cleanup_memory()
    return predictions


def _lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    dp = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        prev = 0
        for index, token_b in enumerate(tokens_b, start=1):
            current = dp[index]
            if token_a == token_b:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = current
    return dp[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_hit_rate(prediction: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    normalized = prediction.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in normalized)
    return hits / len(keywords)


def _aggregate_manual_scores(path: Path | None, mode: str) -> dict[str, Any]:
    if not path or not path.exists():
        return {}

    rows = read_jsonl(path)
    if mode == "anime":
        fields = [
            "style_consistency",
            "fluency",
            "instruction_alignment",
        ]
        summary: dict[str, float] = {}
        for prefix in ("base", "adapter"):
            for field in fields:
                values = [
                    row[f"{prefix}_{field}"]
                    for row in rows
                    if row.get(f"{prefix}_{field}") is not None
                ]
                if values:
                    summary[f"{prefix}_{field}"] = round(sum(values) / len(values), 4)
        return summary

    if mode == "faq":
        summary = {}
        for prefix in ("base", "adapter"):
            values = [
                row[f"{prefix}_correct"]
                for row in rows
                if row.get(f"{prefix}_correct") is not None
            ]
            if values:
                summary[f"{prefix}_correct_rate"] = round(sum(values) / len(values), 4)
        return summary
    return {}


def _load_style_cues(path: Path | None) -> dict[str, list[str]]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_anime_outputs(
    prompts: list[dict[str, Any]],
    base_predictions: list[str],
    adapter_predictions: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for record, base_prediction, adapter_prediction in zip(
        prompts, base_predictions, adapter_predictions
    ):
        rows.append(
            {
                "id": record["id"],
                "trait": record.get("trait"),
                "user": record["user"],
                "base_response": base_prediction,
                "adapter_response": adapter_prediction,
                "base_style_consistency": None,
                "adapter_style_consistency": None,
                "base_fluency": None,
                "adapter_fluency": None,
                "base_instruction_alignment": None,
                "adapter_instruction_alignment": None,
                "notes": "",
            }
        )
    return rows


def _build_faq_manual_sheet(
    prompts: list[dict[str, Any]],
    base_predictions: list[str],
    adapter_predictions: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for record, base_prediction, adapter_prediction in zip(
        prompts, base_predictions, adapter_predictions
    ):
        rows.append(
            {
                "id": record["id"],
                "category": record.get("category"),
                "user": record["user"],
                "reference": record.get("assistant"),
                "base_prediction": base_prediction,
                "adapter_prediction": adapter_prediction,
                "base_correct": None,
                "adapter_correct": None,
                "notes": "",
            }
        )
    return rows


def _classify_error(rouge_score: float, keyword_score: float) -> str:
    if rouge_score < 0.15 and keyword_score == 0:
        return "答非所问"
    if keyword_score < 0.5:
        return "关键信息缺失"
    return "模板化回复过强"


def _build_error_analysis(
    prompts: list[dict[str, Any]],
    predictions: list[str],
    output_path: Path,
) -> None:
    rows = []
    for prompt, prediction in zip(prompts, predictions):
        rouge_score = rouge_l_f1(prediction, prompt.get("assistant", ""))
        keyword_score = keyword_hit_rate(prediction, prompt.get("keywords", []))
        rows.append(
            {
                "id": prompt["id"],
                "category": prompt.get("category", "unknown"),
                "user": prompt["user"],
                "reference": prompt.get("assistant", ""),
                "prediction": prediction,
                "rouge_l": rouge_score,
                "keyword_hit_rate": keyword_score,
                "error_type": _classify_error(rouge_score, keyword_score),
            }
        )

    hardest = sorted(rows, key=lambda row: (row["rouge_l"], row["keyword_hit_rate"]))[:10]
    counter = Counter(row["error_type"] for row in rows)

    lines = [
        "# FAQ Error Analysis",
        "",
        "## Error Distribution",
        "",
    ]
    for label, count in counter.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Hard Cases", ""])
    for row in hardest:
        lines.append(f"### {row['id']} | {row['error_type']}")
        lines.append(f"- User: {row['user']}")
        lines.append(f"- Reference: {row['reference']}")
        lines.append(f"- Prediction: {row['prediction']}")
        lines.append(f"- ROUGE-L: {row['rouge_l']:.4f}")
        lines.append(f"- Keyword Hit Rate: {row['keyword_hit_rate']:.4f}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    compare_base: bool = True,
) -> dict[str, Any]:
    prompts = _load_prompts(config)
    base_predictions = (
        _collect_predictions(config, prompts, checkpoint_path=None) if compare_base else []
    )
    adapter_predictions = _collect_predictions(config, prompts, checkpoint_path=checkpoint_path)

    output_dir = Path(checkpoint_path) / "eval_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "task_name": config.task_name,
        "mode": config.evaluation.mode,
        "num_examples": len(prompts),
    }

    if config.evaluation.mode == "anime":
        judge_rows = _build_anime_outputs(
            prompts=prompts,
            base_predictions=base_predictions or [""] * len(prompts),
            adapter_predictions=adapter_predictions,
        )
        write_jsonl(output_dir / "anime_judge_sheet.jsonl", judge_rows)
        style_cues = _load_style_cues(config.evaluation.style_cues_path)
        if style_cues:
            if base_predictions:
                summary["base_style_cue_hit_rate"] = round(
                    sum(
                        keyword_hit_rate(
                            prediction,
                            style_cues.get(prompt.get("trait", ""), []),
                        )
                        for prompt, prediction in zip(prompts, base_predictions)
                    )
                    / len(prompts),
                    4,
                )
            summary["adapter_style_cue_hit_rate"] = round(
                sum(
                    keyword_hit_rate(
                        prediction,
                        style_cues.get(prompt.get("trait", ""), []),
                    )
                    for prompt, prediction in zip(prompts, adapter_predictions)
                )
                / len(prompts),
                4,
            )
        summary["manual_scores"] = _aggregate_manual_scores(
            config.evaluation.manual_score_path,
            mode="anime",
        )
    else:
        manual_rows = _build_faq_manual_sheet(
            prompts=prompts,
            base_predictions=base_predictions or [""] * len(prompts),
            adapter_predictions=adapter_predictions,
        )
        write_jsonl(output_dir / "faq_manual_sheet.jsonl", manual_rows)
        if base_predictions:
            summary["base_rouge_l"] = round(
                sum(
                    rouge_l_f1(prediction, prompt.get("assistant", ""))
                    for prompt, prediction in zip(prompts, base_predictions)
                )
                / len(prompts),
                4,
            )
        summary["adapter_rouge_l"] = round(
            sum(
                rouge_l_f1(prediction, prompt.get("assistant", ""))
                for prompt, prediction in zip(prompts, adapter_predictions)
            )
            / len(prompts),
            4,
        )
        if base_predictions:
            summary["base_keyword_hit_rate"] = round(
                sum(
                    keyword_hit_rate(prediction, prompt.get("keywords", []))
                    for prompt, prediction in zip(prompts, base_predictions)
                )
                / len(prompts),
                4,
            )
        summary["adapter_keyword_hit_rate"] = round(
            sum(
                keyword_hit_rate(prediction, prompt.get("keywords", []))
                for prompt, prediction in zip(prompts, adapter_predictions)
            )
            / len(prompts),
            4,
        )
        summary["manual_scores"] = _aggregate_manual_scores(
            config.evaluation.manual_score_path,
            mode="faq",
        )
        _build_error_analysis(
            prompts,
            adapter_predictions,
            output_dir / "faq_error_analysis.md",
        )

    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
