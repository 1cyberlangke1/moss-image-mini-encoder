# MOSS-Image 小文本编码器 (MOSS Image Mini Encoder)

> 🚀 **Hugging Face 权重与适配器发布页**：[https://huggingface.co/cyberlangke/moss-image-mini-encoder](https://huggingface.co/cyberlangke/moss-image-mini-encoder)

本项目使用 **Qwen3.5 (0.8B / 2B / 4B)** 替换 MOSS-Image 原生庞大的 **9B** 文本编码器。通过轻量级逐 token 线性投影层，将较小规模语言模型的隐状态直接映射至 4096 维特征空间并输入 DiT 主干。语言模型在推理时全程冻结，仅作为条件特征提取器，可在保证语义表达的同时显著降低显存开销。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-moss--image--mini--encoder-ffd21e.svg)](https://huggingface.co/cyberlangke/moss-image-mini-encoder)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-%E2%89%A50.42.0-4b8bbe.svg)](https://github.com/comfyanonymous/ComfyUI)

---

## 规格与配置对照

所有档位均基于相同的 9B 教师模型及 6 层特征抽头（第 2 / 4 / 8 / 12 / 16 层与最后一层）训练。用户可根据实际显存预算选择合适的模型规格与量化版本：

| 学生模型 | 权重格式 | 模型权重体积 | 适配器体积 | 适配器显存估算 | 说明 |
|---|---|---:|---:|---:|---|
| **0.8B** | bf16 | 1.63 GB | 48.1 MB | ~1.80 GB | 适合显存有限但需轻量验证场景 |
| **0.8B** | int8 | 1.15 GB | 48.1 MB | — | 基础量化版 |
| **0.8B** | w4a8 | 0.94 GB | 48.1 MB | ~1.02 GB | **8GB 显卡推荐**，极低显存占用 |
| **2B** | bf16 | 4.24 GB | 96.1 MB | — | 均衡型全精度 |
| **2B** | int8 | 2.90 GB | 96.1 MB | — | 均衡型量化 |
| **2B** | w4a8 | 2.32 GB | 96.1 MB | ~2.50 GB | 性能与资源消耗折中档位 |
| **4B** | w4a8 | 3.75 GB | 120.1 MB | ~3.75 GB | 高保真首选档位 |

> ⚠️ **匹配注意事项**：不同尺寸的学生模型（0.8B / 2B / 4B）特征维度不同，**不可跨尺寸混用**（会报维度错误）；同尺寸下的不同量化格式（如 bf16 / int8 / w4a8）虽能正常加载，但交叉使用可能导致特征轻微漂移、**可能会损失生成效果**，建议优先搭配对应量化版本的专属适配器。

### 选型建议

- **8GB 显存设备**：推荐选择 **0.8B w4a8**。文本编码器仅占用约 1GB 显存，能与 DiT 模型常驻显存稳定运行。
- **追求高生成质量**：推荐选择 **4B w4a8**。
- **平衡型配置**：推荐选择 **2B w4a8**。

*(注：详细的生成质量测试与端到端耗时基准正在重新评测中，后续更新。)*

---

## 在 ComfyUI 中使用

### 1. 安装自定义节点

进入 ComfyUI 根目录，将本仓库克隆至 `custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1cyberlangke1/moss-image-mini-encoder.git
```

### 2. 放置模型与适配器文件

从 [Hugging Face 仓库](https://huggingface.co/cyberlangke/moss-image-mini-encoder) 下载对应权重并放置于以下路径：

| 目标目录 | 放置内容 |
|---|---|
| `ComfyUI/models/text_encoders/` | Qwen3.5 学生模型权重（如 `qwen3.5-0.8b-w4a8.safetensors`） |
| `ComfyUI/models/moss_encoder_adapters/` | 对应规格的投影适配器 `.safetensors` 文件 |

### 3. 工作流节点连接

在 ComfyUI 工作流中使用 `MossCLIPLoader` 节点替代官方的 `CLIPLoader`：

```
MossCLIPLoader(clip_name, type="moss_image", adapter)
  └── CLIP 输出 ──> CLIPTextEncode（官方正向/负向节点） ──> KSampler
```

- `MossCLIPLoader` 仅比原生节点多出 `adapter` 选项，输出的 CLIP 接口完全兼容官方标准。
- 当 `adapter` 选为 `none` 时，不启用投影，行为与官方原生加载一致。
- 详细节点参数与说明可参考 [`src/comfyui_node`](src/comfyui_node/README.md)。

---

## 为什么有效

1. **同构分词与严格空间对齐**：Qwen3.5 全系列（0.8B 至 9B）共享完全一致的词表和分词逻辑。同一个提示词在不同模型中具备完全相同的 Token 切分与位置对应，使得逐 Token 投影在空间拓扑上天然具备严格的对齐基础。
2. **文本编码器作为特征提取器**：MOSS-Image 的扩散模型通过 Cross-Attention 接收 4096 维条件向量。文本编码器在推理时全程冻结，其核心任务是将提示词转化为高维语义特征向量，DiT 重点关注各 Token 的语义表征方向。
3. **标准化空间闭式解**：语言模型隐藏状态在不同通道上的方差跨越数个数量级。通过在标准化特征空间内求解正则化最小二乘（Ridge 闭式解），有效平衡了各维度的特征权重，能够在保证泛化性的同时获取全局最优映射。

---

## 架构特性与已知限制

- **固定 6 层特征抽头**：当前版本抽取指定 6 层隐藏状态，如迁移到其他基础模型需重新适配层结构。
- **专用于 MOSS-Image**：当前投影层输出固定为 4096 维，用于对齐 MOSS-Image 的 DiT 条件输入，不适用于其他维度的扩散模型。
- **序列长度**：训练集提示词长度覆盖中短篇幅（最长约 3200 token），超长提示词可能超出拟合分布。

---

## 子模块文档

| 模块 | 路径 | 内容概要 |
|---|---|---|
| 训练与编码 | [`src/moss_small_encoder`](src/moss_small_encoder/README.md) | 语料清洗、特征提取、Ridge 线性解求解与权重导出 |
| ComfyUI 节点 | [`src/comfyui_node`](src/comfyui_node/README.md) | 节点参数解析、适配器加载机制与显存调度 |
| 工具脚本 | [`src/scripts`](src/scripts/README.md) | 官方权重下载校验与模型量化脚本 |
| 模型配置 | [`models`](models/README.md) | 官方底模、教师权重及 LoRA 规格说明 |

---

## 致谢

- [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) — 投影层训练与特征处理参考
- [moss-image-smaller-encoder](https://github.com/bakapotatolord/moss-image-smaller-encoder) — 适配器结构与命名规范参考
- [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) — MOSS 扩散模型与布局提示词规范
- [Qwen3.5](https://huggingface.co/Qwen) — 基础语言模型

---

## 许可证

本项目代码与适配器权重遵循 [Apache License 2.0](LICENSE) 许可协议。第三方组件声明请参阅 [NOTICE](NOTICE)。
