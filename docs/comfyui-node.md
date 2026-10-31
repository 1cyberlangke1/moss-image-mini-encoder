# ComfyUI 自定义节点

本项目提供专用的 ComfyUI 自定义节点：**`MossCLIPLoader`**。用于替换 ComfyUI 原生 `CLIPLoader`，在加载 Qwen3.5 学生文本编码器的同时，挂载匹配的线性投影适配器。

```
MossCLIPLoader(学生权重, type="moss_image", adapter) ──> CLIP ──> CLIPTextEncode（官方节点） ──> KSampler
```

该节点输出标准的 CLIP 格式对象，与下游官方原生节点无缝兼容。

---

## 安装说明

把仓库的 `src/comfyui_node` 放进 `custom_nodes`（放**这一层**，不是整个仓库）：

```powershell
New-Item -ItemType Junction `
  -Path "C:\Users\<你>\Documents\ComfyUI\custom_nodes\moss-08b-adapter" `
  -Target "D:\path\to\moss-image-mini-encoder\src\comfyui_node"
```

或直接复制 `src/comfyui_node` 成 `custom_nodes/moss-08b-adapter`，然后重启 ComfyUI。

整仓放进 `custom_nodes` 会报 `FileNotFoundError: ...\moss-image-mini-encoder\__init__.py`，节点不会注册。

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
| `clip_name` | 必填下拉 | `models/text_encoders` 目录下的学生模型文件名。如使用官方 9B 原生模型可直接使用官方 `CLIPLoader`。 |
| `type` | 必填选项 | 固定为 `moss_image`。 |
| `adapter` | 必填下拉 | 选择 `models/moss_encoder_adapters/` 中的适配器文件名；选为 `none` 时不进行投影，等价于原生行为。 |
| `device` | 可选（高级） | 可选 `default` 或 `cpu`。默认由 ComfyUI 管理显存；若显存极其紧张，可设为 `cpu` 将文本编码器常驻于内存。 |

> ⚠️ **匹配建议**：不同尺寸的学生模型（0.8B / 2B / 4B）特征维度不同，**不可跨尺寸混用**；同尺寸下不同量化格式（如 bf16 / int8 / w4a8）具有相同的维度可正常加载，但交叉使用可能导致特征轻微漂移、可能会影响生成效果，建议优先使用同量化版本训练的专属适配器。

---

## 运行机制

节点内部遵循与训练端一致的前向流程：

1. **底模加载**：调用 ComfyUI 基础加载逻辑载入学生模型权重。
2. **多层特征截取**：提取模型指定的 6 个特征层（第 2 / 4 / 8 / 12 / 16 层与最后一层）的隐藏状态。
3. **特征拼接**：按顺序拼接各层特征向量（对应尺寸拼接后维度分别为 6144 / 12288 / 15360 维）。
4. **线性变换与归一化**：对特征进行均值与方差标准化后，与权重矩阵相乘并重标定至 4096 维。
5. **首 Token 校准**：使用适配器中打包的 `sink` 向量校准第 0 个 token 的分布。
6. **环境复位**：前向完成后恢复模型默认配置，不影响其它流程调用该模型。

---

## 显存与运行建议

- **显存敏感环境**：优先使用 **0.8B w4a8** 档位，体积与显存占用最小。
- **设备调度**：一般情况下保持 `device=default` 即可，ComfyUI 会在文本编码完成后自动释放显存以供 DiT 采样；若在极高分辨率下显存不足，可将 `device` 设置为 `cpu`。
