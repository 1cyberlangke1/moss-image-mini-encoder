# ComfyUI 节点（当前为诊断用途）

本目录提供的是**诊断用**节点，不是给终端用户直接使用的成品节点。

## 现在有什么

| 节点 | 作用 |
|---|---|
| `MossCondLoader` | 读取预先算好的条件张量文件（`cond` 键，`[T, 4096]` 或 `[1, T, 4096]`），包成 CONDITIONING |
| `MossCondDumper` | 把 CONDITIONING 导出成 safetensors，用于跨实现对齐校验 |

配合 `tmp/make_cond.py` 使用：那里在项目侧跑 0.8B + 投影算出 cond，本节点只负责读文件。
这样 ComfyUI 侧不需要装 transformers，也不需要跑编码器。

## 为什么还不是成品

终端用户需要的是一个**一步到位**的节点：输入文本 → 节点内部完成分词、编码、投影 →
输出 CONDITIONING。目前缺的部分：

1. **适配器加载**：解析 `potatoforge_moss_encoder_projection_v1` 格式
   （`weight` / `mean_in` / `std_in` / `mean_out` / `std_out`，另可选 `mlp.*` 残差层）
2. **模型接入**：用 ComfyUI 自带的 `qwen35_08b` 预设加载 0.8B，并取 6 个抽头
   （第 2、4、8、12、16 层与最后一层），按顺序拼成 6144 维
3. **模板与对齐**：必须复刻 MOSS 写死的对话模板
   `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n`，
   pad id `248044`，最长 4096 token；否则 token id 与训练时不一致，投影失效
4. **sink 处理**：每条 prompt 第 0 个 token 是 attention sink（方向恒定、范数极大），
   训练时已剔除；推理时必须用 `sink.npy` 的均值替换，不能保留原始值
5. **数值路径**：投影在 CPU fp32 上算、输出 fp16，与训练导出时一致

## 参考实现

参考项目 `moss-image-smaller-encoder` 的做法可以照搬：包装 ComfyUI 的 CLIP 对象，
覆写 `encode_from_tokens` / `encode_from_tokens_scheduled` / `clone`，
并要求 `clip_name == "qwen35_4b"`（我们这里是 `qwen35_08b`）。

## 安装

把本目录链接或复制到 `ComfyUI/custom_nodes/` 下，重启 ComfyUI。
