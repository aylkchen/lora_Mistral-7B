# 基于 Mistral-7B 的参数高效微调实验项目

这是一个围绕 `LoRA + SFT` 搭建的统一微调实验仓库，目标不是做一个聊天 demo，而是把 `数据整理 -> chat template 构造 -> 参数高效训练 -> 推理验证 -> 效果评测 -> 对比实验` 这条链路完整跑通，并且能够直接用于简历和面试讲述。

项目包含两个场景：

- `动漫人格对话`：保留原始 notebook 的主线成果，用来证明你做过 LoRA/SFT 训练。
- `FAQ/客服助手`：补一个更接近业务场景的分支，避免项目显得过于 toy。

## 技术栈

`Python`、`PyTorch`、`Transformers`、`PEFT(LoRA)`、`TRL(SFTTrainer)`、`Hugging Face Datasets`、`YAML`、`Jupyter Notebook`

## 项目亮点

- 将原先集中在 [Untitled.ipynb](./Untitled.ipynb) 的 LoRA 微调流程拆分为脚本化工程：
  - [train.py](./train.py)
  - [evaluate.py](./evaluate.py)
  - [configs](./configs)
- 统一数据格式为聊天监督微调样本：

```json
{"system": "...", "user": "...", "assistant": "..."}
```

- 同时支持：
  - `Mistral-7B-Instruct-v0.1` 的正式训练配置
  - `r=8 / r=16` 的 LoRA 对比实验配置
  - 本地 `tiny-random-Llama` smoke 配置，方便先验证训练和评测链路
- 评测层面覆盖：
  - 动漫场景：生成 `judge sheet`，用于人工打分 `风格一致性 / 回答流畅度 / 指令贴合度`
  - FAQ 场景：计算 `ROUGE-L` 和 `keyword hit rate`，并生成错误案例分析

## 仓库结构

```text
.
|-- Untitled.ipynb
|-- train.py
|-- evaluate.py
|-- configs/
|   |-- anime_lora.yaml
|   |-- anime_lora_r8.yaml
|   |-- faq_lora.yaml
|   |-- faq_lora_r8.yaml
|   |-- smoke_anime.yaml
|   `-- smoke_faq.yaml
|-- data/
|   |-- anime/
|   |   |-- splits/
|   |   |-- eval_prompts.jsonl
|   |   `-- style_cues.json
|   `-- faq/
|       |-- splits/
|       |-- heldout.jsonl
|       `-- seed_faq.jsonl
|-- lora_lab/
|   |-- config.py
|   |-- data.py
|   |-- modeling.py
|   |-- training.py
|   `-- evaluation.py
|-- outputs/
|   |-- smoke_anime_tiny/
|   `-- smoke_faq_tiny/
`-- scripts/
    `-- bootstrap_demo_data.py
```

## 数据设计

### 1. 动漫人格对话

- 样本规模：`744` 条
- 人格类别：`6` 类
  - `tsundere`
  - `yandere`
  - `himedere`
  - `genki`
  - `moe`
  - `bakadere`
- 划分方式：`train / val / test = 8 : 1 : 1`
- 额外资产：
  - `50` 条评测 prompt
  - `style_cues.json`，用于风格提示词命中率的自动代理评测

### 2. FAQ / 客服助手

- 种子 FAQ：`360` 条
- 扩写后 SFT 样本：`1080` 条
- Held-out 问题：`100` 条
- 场景覆盖：
  - 物流
  - 退货
  - 保修
  - 支付
  - 账号
  - 售后支持

如果后续要替换成自己的真实数据，只需要保持 `system / user / assistant` 字段不变，然后修改对应配置里的 `dataset_path` 即可。

## 训练与评测

### 1. 生成数据

```bash
python scripts/bootstrap_demo_data.py
```

### 2. 训练动漫人格 LoRA

```bash
python train.py --config configs/anime_lora.yaml
```

### 3. 训练 FAQ / 客服助手 LoRA

```bash
python train.py --config configs/faq_lora.yaml
```

### 4. 评测动漫人格模型

```bash
python evaluate.py --config configs/anime_lora.yaml --checkpoint outputs/anime_mistral_lora
```

输出：

- `anime_judge_sheet.jsonl`
- `evaluation_summary.json`

### 5. 评测 FAQ 模型

```bash
python evaluate.py --config configs/faq_lora.yaml --checkpoint outputs/faq_mistral_lora
```

输出：

- `faq_manual_sheet.jsonl`
- `faq_error_analysis.md`
- `evaluation_summary.json`

## 对比实验建议

这个仓库已经把对比实验入口预留好了，建议服务器上至少补这三组：

1. `基座模型 vs LoRA 模型`
2. `r=8 vs r=16`
3. `动漫场景 vs FAQ 场景`

可直接使用：

- [configs/anime_lora.yaml](./configs/anime_lora.yaml)
- [configs/anime_lora_r8.yaml](./configs/anime_lora_r8.yaml)
- [configs/faq_lora.yaml](./configs/faq_lora.yaml)
- [configs/faq_lora_r8.yaml](./configs/faq_lora_r8.yaml)

## 当前结果

### 原始 notebook 基线

这是原项目在 [Untitled.ipynb](./Untitled.ipynb) 中已经得到的真实结果：

| 任务 | 模型/配置 | 训练结果 | 评测/结论 |
| --- | --- | --- | --- |
| 动漫人格对话 | `Mistral-7B-Instruct-v0.1 + LoRA(r=16, alpha=32, dropout=0.05)` | `train loss = 0.9091`，训练 `141` steps，可训练参数 `13,631,488 / 7,255,363,584 = 0.1879%` | 对同一问题进行多人格生成时，输出风格已经能体现 `tsundere / yandere / genki / moe` 等差异 |

### 仓库内 smoke test

这些结果主要用于验证训练和评测脚本已经打通，不代表最终 Mistral-7B 实验上限：

| 任务 | 配置 | 训练结果 | 评测/产物 |
| --- | --- | --- | --- |
| 动漫人格 smoke | `tiny-random-Llama + LoRA(r=4)` | `train loss = 10.2802`，`val loss = 10.2838`，可训练参数占比 `0.0991%` | 生成 `anime_judge_sheet.jsonl` 与 `evaluation_summary.json` |
| FAQ smoke | `tiny-random-Llama + LoRA(r=4)` | `train loss = 10.2638`，`val loss = 10.2652`，可训练参数占比 `0.0991%` | 生成 `faq_manual_sheet.jsonl`、`faq_error_analysis.md` 与 `evaluation_summary.json` |

参考文件：

- [outputs/smoke_anime_tiny/training_summary.json](./outputs/smoke_anime_tiny/training_summary.json)
- [outputs/smoke_anime_tiny/eval_outputs/evaluation_summary.json](./outputs/smoke_anime_tiny/eval_outputs/evaluation_summary.json)
- [outputs/smoke_faq_tiny/training_summary.json](./outputs/smoke_faq_tiny/training_summary.json)
- [outputs/smoke_faq_tiny/eval_outputs/evaluation_summary.json](./outputs/smoke_faq_tiny/eval_outputs/evaluation_summary.json)

## 面试

`基于 Mistral-7B-Instruct 完成动漫人格对话和 FAQ 客服助手两个场景的参数高效微调实验，使用 PEFT 对注意力层 q/k/v/o 投影注入 LoRA，通过 TRL 的 SFTTrainer 完成监督微调，并搭建了从数据整理、chat template 构造、训练、推理到人工评测/自动评测的完整闭环；其中动漫场景数据 744 条、6 类人格，FAQ 场景包含 360 条种子问答与 1080 条 SFT 样本，原始实验可训练参数占比约 0.188%，验证了在低成本训练条件下实现风格化生成和窄域问答微调的可行性。`

## 简历关键词

- `LoRA`
- `SFT`
- `PEFT`
- `TRL`
- `参数高效微调`
- `指令微调`
- `大模型训练经验`
- `评测与对比实验`

## 说明

- 当前仓库默认保留了原始 [Untitled.ipynb](./Untitled.ipynb)，方便你回看最初的实验过程。
- `train.py` 已经处理了 Windows 下 `TRL` 的 UTF-8 兼容问题，本地可先 smoke，正式实验建议在 Linux/CUDA 服务器上运行。
- 如果你服务器上的基础模型路径不是 Hugging Face 名称，而是本地绝对路径，只需要修改对应 YAML 里的 `base_model`。
