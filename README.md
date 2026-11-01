# MOSS Image Mini Encoder

> 🚀 **Hugging Face 权重与适配器发布页**：[https://huggingface.co/cyberlangke/moss-image-mini-encoder](https://huggingface.co/cyberlangke/moss-image-mini-encoder)

MOSS-Image 官方默认的 9B 文本编码器体积较大，在消费级显卡上运行时显存压力明显。本项目使用较小尺寸的 **Qwen3.5 (0.8B / 2B / 4B)** 替换原生 9B 文本编码器，通过一层轻量的逐 token 线性投影层，将较小规模语言模型的隐藏状态直接映射至 4096 维并输入 DiT。语言模型在推理时全程冻结，仅作为特征提取器，在降低显存开销的同时保留生成能力。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-moss--image--mini--encoder-ffd21e.svg)](https://huggingface.co/cyberlangke/moss-image-mini-encoder)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-%E2%89%A50.42.0-4b8bbe.svg)](https://github.com/comfyanonymous/ComfyUI)

---

## 规格与配置对照

所有档位均基于相同的 9B 教师模型及 6 层特征抽头（第 2 / 4 / 8 / 12 / 16 层与最后一层）训练。可根据实际显存预算选择合适的模型规格与量化版本：

| 学生模型 | 权重格式 | 模型权重体积 | 适配器体积 | 学生+适配器显存估算 | 说明 |
|---|---|---:|---:|---:|---|
| **0.8B** | bf16 | 1.63 GB | 48.1 MB | ~1.80 GB | 全精度版，适合对比验证 |
| **0.8B** | int8 | 1.15 GB | 48.1 MB | ~1.25 GB | 基础量化版 |
| **0.8B** | w4a8 | 0.94 GB | 48.1 MB | ~1.02 GB | 显存占用最小档位 |
| **2B** | bf16 | 4.24 GB | 96.1 MB | ~4.50 GB | 全精度版 |
| **2B** | int8 | 2.90 GB | 96.1 MB | ~3.10 GB | 基础量化版 |
| **2B** | w4a8 | 2.32 GB | 96.1 MB | ~2.50 GB | 速度与显存均衡档位 |
| **4B** | w4a8 | 3.75 GB | 120.1 MB | ~3.87 GB | 效果最接近 9B 档位 |

> ⚠️ **匹配注意事项**：不同尺寸的学生模型（0.8B / 2B / 4B）特征维度不同，**不可跨尺寸混用**（会报维度错误）；同尺寸下的不同量化格式（如 bf16 / int8 / w4a8）具有相同的维度可正常加载，但交叉使用可能导致特征轻微漂移、可能会影响生成效果，建议优先搭配对应量化版本的专属适配器。

### 选型建议

- **显存优先**：0.8B w4a8。
- **平衡首选**：2B w4a8。
- **注重细节**：4B w4a8。

---

## 实测效果

固定 seed 42、4 步 LoRA、cfg 1.0、分辨率 1024×1024，四档跑同一条**只给角色名**的提示词：

> 蔚蓝档案的鬼方カヨコ蹲在巷子里喂流浪猫

![四档对比](docs/images/compare/prompt-02.jpg)

| 档位 | 生成结果 | cos vs 9B | 端到端热态 |
|---|---|---:|---:|
| 9B w4a8 | 白发黑挑染、红瞳、双角、光环、黑外套——角色正确 | 1.0000 | 33.1 s |
| 4B w4a8 | 有角与光环，但蓝瞳、无挑染——落在作品域内，认错人 | 0.8638 | 23.6 s |
| 2B w4a8 | 白发蓝瞳的普通少女，没有角也没有光环，与 0.8B 的区别只是换了发色 | 0.8372 | 20.4 s |
| 0.8B bf16 | 金发蓝眼、白衬衫蓝裙——未识别该专名 | 0.8149 | 20.3 s |

**只给品牌名**：

> 蜜雪冰城的雪王站在店门口

![四档对比](docs/images/compare/prompt-04.jpg)

| 档位 | 生成结果 |
|---|---|
| 9B w4a8 | 招牌「蜜雪冰城」、白胖雪人、金冠、红披风、冰淇淋火炬，身上还有竖排字，全对 |
| 4B w4a8 | 以上要素齐全，只少了身上的竖排字 |
| 2B w4a8 | 招牌与金冠正确，但雪人变成真的雪堆、披风变成红围巾、没有火炬 |
| 0.8B bf16 | 招牌写成「𠳨雪冰城」，雪王被画成穿龙袍的白胡子老者 |

只给名称时，小模型会认错人甚至完全认不出——判据是出图；余弦只说明条件向量偏了。

完整评测对比（10 条提示词 × 8 档，含出图细节、余弦对齐与耗时数据）见
[**docs/benchmark.md**](docs/benchmark.md)。

---

## 在 ComfyUI 中使用

### 1. 安装自定义节点

把整个仓库放进 `ComfyUI/custom_nodes/` 下，重启 ComfyUI 即可：

```powershell
git clone https://github.com/1cyberlangke1/moss-image-mini-encoder `
  "C:\Users\<你>\Documents\ComfyUI\custom_nodes\moss-image-mini-encoder"
```

节点实现位于 `src/comfyui_node/`，仓库根的 `__init__.py` 负责把它接进 ComfyUI 喵。

### 2. 放置模型与适配器文件

从 [Hugging Face 仓库](https://huggingface.co/cyberlangke/moss-image-mini-encoder) 下载对应权重并放置于以下路径：

| 目标目录 | 放置内容 |
|---|---|
| `ComfyUI/models/text_encoders/` | Qwen3.5 学生权重：量化版（int8 / w4a8）取 HF 仓库的 `weights/`；bf16 用官方原版 |
| `ComfyUI/models/moss_encoder_adapters/` | 对应规格的投影适配器 `.safetensors` 文件 |

官方权重来源：

| 用途 | 链接 |
|---|---|
| 0.8B bf16 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| 2B bf16 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| 4B 官方原版权重（bf16） | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)；本仓库仅为 4B 训练 w4a8 适配器，不提供 bf16 适配器 |
| 9B 教师（仅重训 / 对照） | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)，或 ComfyUI 可直接用的 [w4a8 量化版](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors) |

### 3. 工作流节点连接

在 ComfyUI 工作流中使用 `MossCLIPLoader` 节点替代官方的 `CLIPLoader`：

```
MossCLIPLoader(clip_name, type="moss_image", adapter)
  └── CLIP 输出 ──> CLIPTextEncode（官方正向/负向节点） ──> KSampler
```

- `MossCLIPLoader` 仅比原生节点多出 `adapter` 选项，输出的 CLIP 接口完全兼容官方标准。
- 当 `adapter` 选为 `none` 时，不启用投影，行为与官方原生加载一致。
- 详细节点参数与说明可参考 [`docs/comfyui-node.md`](docs/comfyui-node.md)。

---

## 为什么有效

1. **同系列统一分词**：Qwen3.5 全系列（0.8B 至 9B）共享相同的词表和分词逻辑。同一个提示词在不同模型中 token 切分与位置严格一致，可直接进行逐 token 线性对齐。
2. **文本特征仅作引导**：MOSS-Image 的扩散模型通过 Cross-Attention 接收 4096 维条件向量。文本编码器推理时保持冻结，核心是提供语义特征方向，不需要语言模型进行生成。
3. **标准化空间闭式解**：语言模型隐藏状态在不同通道上的数值方差差异较大。在标准化特征空间内直接求解正则化最小二乘（Ridge 闭式解），计算快速稳定，且无传统迭代训练容易出现的过拟合问题。

---

## 架构特性与已知限制

- **固定 6 层特征抽头**：当前版本抽取指定 6 层隐藏状态，迁移到其他基础模型需重新适配层结构。
- **专用于 MOSS-Image**：当前投影层输出固定为 4096 维，用于对齐 MOSS-Image 的 DiT 条件输入，不适用于其他维度的扩散模型。
- **序列长度**：训练集提示词长度覆盖中短篇幅（最长约 3200 token），超长提示词上限支持 4096 token。

---

## 子模块文档

| 模块 | 路径 | 内容概要 |
|---|---|---|
| 训练与编码 | [`src/moss_small_encoder`](docs/pipeline.md) | 语料清洗、特征提取、Ridge 线性解求解与权重导出 |
| ComfyUI 节点 | [`src/comfyui_node`](docs/comfyui-node.md) | 节点参数解析、适配器加载机制与显存调度 |
| 工具脚本 | [`src/scripts`](docs/scripts.md) | 官方权重下载校验与模型量化脚本 |
| 模型配置 | [`models`](docs/models.md) | 官方底模、教师权重及 LoRA 规格说明 |
| 效果与速度评测 | [`docs/benchmark`](docs/benchmark.md) | 10 条提示 × 8 档出图效果、余弦相似度与耗时对比 |

---

## 致谢

- [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) — 投影层训练与特征处理参考
- [moss-image-smaller-encoder](https://github.com/bakapotatolord/moss-image-smaller-encoder) — 适配器结构与命名规范参考
- [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) — MOSS 扩散模型与布局提示词规范
- Qwen3.5 — 基础语言模型：[0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) / [2B](https://huggingface.co/Qwen/Qwen3.5-2B) / [4B](https://huggingface.co/Qwen/Qwen3.5-4B) / [9B](https://huggingface.co/Qwen/Qwen3.5-9B)

---

## 许可证

本项目代码与适配器权重遵循 [Apache License 2.0](LICENSE) 许可协议。第三方组件声明请参阅 [NOTICE](NOTICE)。
