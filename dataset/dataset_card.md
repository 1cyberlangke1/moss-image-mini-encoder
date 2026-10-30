# 数据集说明 (Dataset Card)

本项目用于训练 Qwen3.5 较小规模文本编码器（0.8B / 2B / 4B）对齐 MOSS-Image 9B 教师模型特征的逐 Token 投影层。

---

## 构建方式

```bash
# 下载依赖与分词器
python src/scripts/download_models.py

# 执行语料清洗、分词与分桶构建
python -m moss_small_encoder.build --config src/configs/dataset_v1.yaml
```

所有外部数据源均锁定固定的 Commit SHA，结合固定随机种子保证数据构建过程的严格可复现性。

---

## 来源与许可信息

| 来源标识 | 样本条数 | 许可协议 | 原始地址 |
| --- | ---: | --- | --- |
| `goku` | 6,676 | CC BY 4.0 | https://huggingface.co/datasets/Goku-OpenLab/gpt-image-2-prompts-datasets |
| `gustavosta` | 8,000 | 未标注（源数据集卡片标注为 unknown） | https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts |
| `jtatman` | 12,800 | 未标注（源自 Civitai 社区语料） | https://huggingface.co/datasets/jtatman/stable-diffusion-prompts-stats-full-uncensored |
| `moss_official` | 20 | Apache-2.0 | https://github.com/OpenMOSS/MOSS-Image |
| `n714` | 4,000 | MIT | https://github.com/n714/stable-diffusion-prompt-library-zh_CN |
| `shuixian` | 4,000 | Attribution Required License v1.0 | https://github.com/BaYue-SYJ/shuixian-prompts |
| `synth_layout` | 3,000 | 本仓库自生成 | 遵循 MOSS-Image 布局规范合成 |

### 署名声明

> 本项目使用了「水仙的AI提示词」（https://github.com/BaYue-SYJ/shuixian-prompts）的中文提示词数据，该数据按 Attribution Required License v1.0 协议使用并进行署名。

---

## 数据集已知特性

- **长度约束**：超过 4096 Token 的样本整条剔除，与模型实际推理端上限保持一致。
- **语种与翻译**：`goku` 包含部分机器翻译中文内容，主要用于扩充多语言词汇覆盖。
- **布局格式**：`synth_layout` 由模板确定性生成，用于覆盖 MOSS-Image 的特定排版语法。
- **内容标注**：采用源数据原生标签与关键词启发式规则结合进行分层抽样。
