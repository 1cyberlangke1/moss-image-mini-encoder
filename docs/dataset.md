# 数据集说明 (Dataset Card)

本项目用于训练 Qwen3.5 学生文本编码器（0.8B / 2B / 4B）对齐 MOSS-Image 9B 教师模型特征的逐 Token 线性投影层。

---

## 构建方式

```bash
# 下载依赖与分词器
python src/scripts/download_models.py

# 让 Python 找得到 src 布局里的包（二选一）
pip install -e .
# export PYTHONPATH=src    # PowerShell: $env:PYTHONPATH="src"

# 执行语料清洗、分词与分桶构建
python -m moss_small_encoder.build --config src/configs/dataset_v1.yaml
```

外部数据源均锁定固定 Commit SHA，配合固定随机种子确保构建过程可严格复现。

---

## 来源与许可信息

| 来源标识 | 样本条数 | 许可协议 | 原始地址 |
| --- | ---: | --- | --- |
| `goku` | 6,676 | CC BY 4.0 | https://huggingface.co/datasets/Goku-OpenLab/gpt-image-2-prompts-datasets |
| `gustavosta` | 8,000 | 未标注 | https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts |
| `jtatman` | 12,800 | 未标注（源自 Civitai 社区语料） | https://huggingface.co/datasets/jtatman/stable-diffusion-prompts-stats-full-uncensored |
| `moss_official` | 20 | Apache-2.0 | https://github.com/OpenMOSS/MOSS-Image |
| `n714` | 4,000 | MIT | https://github.com/n714/stable-diffusion-prompt-library-zh_CN |
| `shuixian` | 4,000 | Attribution Required License v1.0 | https://github.com/BaYue-SYJ/shuixian-prompts |
| `synth_layout` | 3,000 | 本仓库自生成 | 遵循 MOSS-Image 布局规范合成 |
| `hard_cases` | 3,840 | 本仓库自生成（LLM 合成） | 由 DeepSeek、Gemini 按固定任务书生成，随仓库分发 |

### 硬 case 语料的来源

`dataset/hard_cases/v1/prompts.jsonl` 由第三方大模型按固定任务书生成：12 个类别（主视觉、封面、海报、菜单、招牌、包装、UI、漫画、长文、群像、户外、信息图）乘 8 种风格，每格 40 条，全中文，`nsfw` 全为 `False`。

数据里只保留模型输出的提示词文本，不含上游模型的权重、服务凭证或对话记录。分发时按本仓库许可提供，使用者仍需自行确认其所用模型服务的条款是否允许此类再分发喵。

### 署名声明

> 本项目使用了「水仙的AI提示词」（https://github.com/BaYue-SYJ/shuixian-prompts）的中文提示词数据，该数据按 Attribution Required License v1.0 协议使用并进行署名。

---

## 数据集特性

- **长度约束**：超过 4096 Token 的样本整条剔除，避免截断产生断句破坏语义；
- **语种与翻译**：`goku` 包含部分机器翻译中文内容，用于扩充多语言词汇覆盖；
- **布局格式**：`synth_layout` 由模板生成，用于覆盖 MOSS-Image 的排版与坐标语法；
- **分层抽样**：采用原生标签与关键词规则结合进行抽样与数据平衡。
