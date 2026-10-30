"""MOSS-Image 条件编码节点：在 ComfyUI 内部跑 Qwen3.5-0.8B 再做线性投影。

数据流：
    文本 → MOSS 对话模板（`<|im_start|>user … <|im_start|>assistant\\n<think>\\n`）
         → Qwen3.5-0.8B 一次前向取 6 个抽头（2/4/8/12/16/最后一层）
         → 逐 token 拼接成 6144 维
         → CPU float32 标准化 + 线性投影 → 4096 维
         → token 0 换回 teacher 实测的 sink 向量
         → CONDITIONING

与训练侧（`moss_small_encoder.encode.student` + `train.ridge`）严格同一口径，
所以同一个适配器在两边算出来的条件应当逐 token 对齐。
"""

from __future__ import annotations

import os

import folder_paths
import torch

from .adapter import adapter_on, load_adapter, load_sink, parse_taps, to_comfy_taps

ADAPTER_FOLDER = "moss_encoder_adapters"
MAX_TOKENS = 4096


def register_adapter_folder() -> str:
    """在 ComfyUI 的模型目录下注册适配器文件夹。

    Returns:
        文件夹绝对路径。目录不存在时创建；重复调用幂等。
    """
    directory = os.path.join(folder_paths.models_dir, ADAPTER_FOLDER)
    os.makedirs(directory, exist_ok=True)
    if ADAPTER_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(ADAPTER_FOLDER, directory)
    return directory


def _adapter_choices() -> list[str]:
    """下拉框选项；目录为空时给一句人话提示，避免空枚举让节点加载失败。"""
    try:
        choices = folder_paths.get_filename_list(ADAPTER_FOLDER)
    except Exception:  # noqa: BLE001 - 节点注册阶段的任何异常都不该拖垮 ComfyUI
        choices = []
    return choices or ["(把适配器放进 models/moss_encoder_adapters)"]


class MossEncoderCondEncode:
    """Qwen3.5-0.8B + 线性投影 → MOSS 条件。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIPLoader 以 type=moss_image 加载的 Qwen3.5-0.8B"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "adapter": (_adapter_choices(), {"tooltip": "models/moss_encoder_adapters 下的适配器"}),
            },
            "optional": {
                "sink": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "sink.npy 的绝对路径；留空则 token 0 也走投影（会偏离训练口径）",
                    },
                ),
                "taps": (
                    "STRING",
                    {"default": "", "tooltip": "覆盖适配器 metadata 里的抽头，如 2,4,8,12,16,-1"},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/MOSS"
    DESCRIPTION = (
        "Encode a prompt with Qwen3.5-0.8B + a learned linear projection instead of the 9B text encoder."
    )

    def encode(self, clip, text, adapter, sink="", taps=""):
        """把一段文本编码成 MOSS 的条件张量。

        Args:
            clip: ComfyUI 的 CLIP 对象，底层是 Qwen3.5-0.8B。
            text: 提示词。内部按 MOSS 固定模板包一层对话样式。
            adapter: 适配器文件名（相对 `models/moss_encoder_adapters`）或绝对路径。
            sink: 可选，teacher 的 token 0 向量（`.npy`）。
            taps: 可选，覆盖适配器里的抽头列表。

        Returns:
            `(CONDITIONING,)`，即 `[[cond, {}]]`，`cond` 为 `[1, T, 4096]` float32。

        Raises:
            ValueError: 适配器/sink 非法、模型层数不够、或抽头与适配器维度对不上。
        """
        resolved = folder_paths.get_full_path(ADAPTER_FOLDER, adapter) or adapter
        loaded = load_adapter(resolved)
        requested = parse_taps(taps) if taps.strip() else loaded["taps"]

        # 通过 CLIP.layer_idx 传抽头：encode_from_tokens 会先 reset 再套用，必须走这条通道。
        # 这里**不 clone**：set_clip_options 最终落在共享的 cond_stage_model 上，
        # clone 隔离不了真正的状态，却要多建一份 patcher 并让 ComfyUI 每次重新走加载判定喵。
        # 共享对象的风险用 try/finally 兜住：无论正常返回还是抛异常都还原原值喵。
        inner = getattr(clip.cond_stage_model, clip.cond_stage_model.clip)
        num_layers = int(inner.num_layers)
        layer_idx = to_comfy_taps(requested, num_layers)
        previous = getattr(clip, "layer_idx", None)
        clip.layer_idx = layer_idx
        try:
            tokens = clip.tokenize(text, thinking=True)
            key = next(iter(tokens))
            if len(tokens[key]) > MAX_TOKENS:
                # 生成侧上限就是 4096，截断要与 9B 一致
                tokens[key] = tokens[key][:MAX_TOKENS]
            # ComfyUI 的执行器已经包了 inference_mode，这里再包一层是给
            # 「节点被别的脚本直接调用」的场景兜底，嵌套没有额外开销喵
            with torch.inference_mode():
                encoded = clip.encode_from_tokens(tokens)
        finally:
            clip.layer_idx = previous
        # return_pooled=False 时只返回条件本身；带 pooled 时是元组
        cond = encoded[0] if isinstance(encoded, (tuple, list)) else encoded

        if cond.ndim != 4:
            raise ValueError(
                f"期望抽头输出 [B, {len(layer_idx)}, T, H]，实际 {tuple(cond.shape)}；"
                "确认 clip 是 Qwen3.5 且抽头列表生效"
            )

        batch, n_taps, n_tok, hidden = cond.shape
        feats = cond.permute(0, 2, 1, 3).reshape(batch, n_tok, n_taps * hidden)[0]
        # 投影跟着 clip 的输出设备走：clip 在 cuda 就全程留 GPU，在 cpu 就留 CPU喵。
        # 之前固定 `.cpu()` 是为了对齐参考实现的 CPU fp32 路径，但 layout JSON 这类
        # 长提示（142 token）在 CPU 上投影要 40 ms，搬到 GPU 只要 3 ms喵。
        device = feats.device
        feats = feats.float()
        if int(feats.shape[-1]) != int(loaded["weight"].shape[0]):
            raise ValueError(
                f"抽头拼接后是 {int(feats.shape[-1])} 维，适配器要 {int(loaded['weight'].shape[0])} 维；"
                f"当前抽头 {requested}（ComfyUI 下标 {layer_idx}）"
            )

        # 训练时每条 prompt 的 token 0 被剔除，所以投影只覆盖 1..T-1
        head = None
        body = feats
        if sink.strip():
            head = load_sink(sink).to(device).float().unsqueeze(0)
            body = feats[1:]

        w = adapter_on(loaded, device)
        z = (body - w["mean_in"]) / w["std_in"]
        # 与训练导出对齐：投影结果先取 fp16 再升回 fp32，保证数值路径一致
        proj = (z @ w["weight"] * w["std_out"] + w["mean_out"]).half().float()
        if not bool(torch.isfinite(proj).all()):
            raise ValueError("投影结果含 NaN/Inf，适配器或输入特征异常")
        out = torch.cat([head, proj], dim=0) if head is not None else proj
        return ([[out.unsqueeze(0).contiguous(), {}]],)


NODE_CLASS_MAPPINGS = {"MossEncoderCondEncode": MossEncoderCondEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossEncoderCondEncode": "MOSS Encode (0.8B + projection)"}
