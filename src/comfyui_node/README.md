# ComfyUI 自定义节点

本项目提供专用的 ComfyUI 自定义节点：**`MossCLIPLoader`**。它用于替换 ComfyUI 原生 `CLIPLoader`，在加载 Qwen3.5 学生文本编码器的同时，可按需挂载匹配的线性投影适配器。

```
MossCLIPLoader(学生权重, type="moss_image", adapter) ──> CLIP ──> CLIPTextEncode（官方节点） ──> KSampler
```

该节点输出标准的 CLIP 格式对象，与下游官方原生节点无缝衔接。

---

## 安装说明

1. 将本仓库目录克隆或链接到 `ComfyUI/custom_nodes/` 目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/1cyberlangke1/moss-image-mini-encoder.git
   ```
2. 重启 ComfyUI 服务。

### 模型与文件放置路径

节点运行需要以下两类模型权重：

| 目录路径 | 放置内容 | 说明 |
|---|---|---|
| `ComfyUI/models/text_encoders/` | Qwen3.5 学生模型权重 | 支持 0.8B / 2B / 4B 的 bf16、int8 或 w4a8 格式 |
| `ComfyUI/models/moss_encoder_adapters/` | 投影适配器文件 | 对应档位的 `.safetensors` 文件 |

*注：`models/moss_encoder_adapters/` 目录在节点初次加载时会自动创建。*

---

## 节点输入参数

| 参数名 | 类型 | 说明 |
|---|---|---|
| `clip_name` | 必填下拉 | `models/text_encoders` 目录下的学生模型文件名。如使用官方 9B 原生模型请直接使用官方 `CLIPLoader`。 |
| `type` | 必填选项 | 固定为 `moss_image`。 |
| `adapter` | 必填下拉 | 选择 `models/moss_encoder_adapters/` 中的适配器文件名；选为 `none` 时不进行投影，等价于原生行为。 |
| `device` | 可选（高级） | 可选 `default` 或 `cpu`。在显存紧张（如 8GB 显卡加载 4B 模型）时可选择 `cpu` 分担显存。 |

> ⚠️ **匹配建议**：不同尺寸的学生模型（0.8B / 2B / 4B）特征维度不同，**不可跨尺寸混用**；同尺寸下不同量化格式（如 bf16 / int8 / w4a8）具有相同的维度可正常加载，但交叉使用可能存在微小特征漂移、**可能会损失生成效果**，建议优先使用同量化版本训练的专属适配器。

---

## 运行机制

节点内部遵循与训练端完全一致的前向流水线：

1. **复用原生加载逻辑**：调用官方 `CLIPLoader` 加载学生模型底模权重。
2. **多层特征抽取**：将 `layer_idx` 动态切至指定 6 个特征层（第 2 / 4 / 8 / 12 / 16 层与最后一层）。
3. **特征拼接**：单次前向后，按顺序拼接各层隐藏状态（拼接后维度对应 6144 / 12288 / 15360 维）。
4. **线性变换与归一化**：执行 `((h - mean_in) / std_in) @ W * std_out + mean_out`，完成维度映射并重归一化。
5. **Token 0 均值对齐**：使用适配器中打包的 `sink` 向量校准首个 token 的表征分布。
6. **状态恢复**：执行完毕后恢复原始 `layer_idx`，不影响其他流程对该模型的共享调用。

*(注：实测节点执行延迟与显存驻留基准数据正在重新评测中，后续更新。)*

---

## 显存与运行建议

- **8GB 显存设备**：
  - 推荐使用 **0.8B w4a8** 档位，文本编码器显存占用约 1.02 GB，可与 MOSS-Image DiT（int8 格式约 5.16 GB）稳定常驻显存运行。
  - 使用 0.8B bf16（约 1.80 GB）时，冷启动显存波动可能接近显存上限。
- **大尺寸模型调度**：
  - **2B w4a8** 与 **4B w4a8** 建议保持 `device=default`，允许 ComfyUI 在文本编码完成后将模型卸载至内存，与 DiT 交替运行。
