"""w4a8 量化权重的加载与 Linear 替换。

Comfy-Org 发布的 MOSS-Image 文本编码器用私有布局 `asym_w4a8_int8`：

    weight            int8 [N, K/2]    每行 K/2 字节打包 16 级 4bit 码（偶数列 = 低半字节）
    weight_codebook   f32  [16]        Lloyd-Max 码表
    weight_s_rel      f8e4m3 [N, K/16] 每组 16 个权重的相对 scale
    weight_s_channel  f32  [N]         每个输出通道的绝对 scale
    comfy_quant       u8   [72]        JSON：{"format": "asym_w4a8_int8", "group_size": 16,
                                        "convrot": true, "convrot_groupsize": 256}

权重在存储前做过 ConvRot 旋转（对 K 轴每 256 个一组做 Hadamard 旋转），所以
**不能**反量化成普通 fp16 权重再喂 `F.linear`——激活侧必须做同样的旋转。
`comfy_kitchen.w4a8_int8_linear` 在融合 kernel 里同时完成旋转、反量化和 GEMM。

本模块只做两件胶水：把张量组装成 `QuantizedTensor`，把 `nn.Linear` 换成
`QuantLinear`。反量化与矩阵乘全部由 comfy_kitchen（pip 包）完成，
这里不含任何量化算法实现。
"""

from __future__ import annotations

import json
from typing import Mapping

import torch
import torch.nn.functional as F
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor

# 由 comfy_quant blob 里的 format 字段标识。
QUANT_FORMAT = "asym_w4a8_int8"
INT8_TENSORWISE = "int8_tensorwise"

# 量化层在 safetensors 里附带的辅助张量后缀；加载时要跳过它们，
# 否则 transformers 的 load_state_dict 会因为形状/类型不符报错。
QUANT_SUFFIXES = (
    "weight_s_rel",
    "weight_s_channel",
    "weight_codebook",
    "weight_scale",
    "comfy_quant",
)


def read_quant_blob(raw: torch.Tensor) -> dict:
    """解析 `comfy_quant` 元数据张量。

    Args:
        raw: uint8 张量，内容是一段以 NUL 补齐的 JSON。

    Returns:
        解析后的字典，至少含 `format`。

    Notes:
        尾部用 b"\\x00" 填充到固定 72 字节，直接 decode 会带出 NUL，先 rstrip。
    """
    text = raw.numpy().tobytes().decode("utf-8").rstrip("\x00")
    return json.loads(text)


def is_asym_w4a8(prefix: str, keys: set[str]) -> bool:
    """判断某个前缀是不是 asym_w4a8_int8 量化层。

    Args:
        prefix: 模块前缀，例如 `model.language_model.layers.0.mlp.gate_proj`。
        keys: safetensors 的全部 key。

    Returns:
        True 表示该前缀由 `weight` + `weight_s_rel` + `weight_s_channel` 构成。
    """
    return f"{prefix}.weight_s_rel" in keys and f"{prefix}.weight_s_channel" in keys


def is_int8_tensorwise(prefix: str, keys: set[str]) -> bool:
    """判断某个前缀是不是 int8_tensorwise 量化（逐行 scale 的 int8）。

    Args:
        prefix: 模块前缀。
        keys: safetensors 的全部 key。

    Returns:
        True 表示该前缀有 `weight_scale` 但没有分组 scale。

    Notes:
        `embed_tokens` 与 `lm_head` 用这个格式，它们不是 matmul 语义，
        不能当 Linear 处理，只做查表反量化。
    """
    return f"{prefix}.weight_scale" in keys and f"{prefix}.weight_s_rel" not in keys


def build_asym_w4a8(
    prefix: str,
    tensors: Mapping[str, torch.Tensor],
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> QuantizedTensor:
    """把一层 safetensors 张量组装成 comfy_kitchen 的量化张量。

    Args:
        prefix: 模块前缀，例如 `...layers.0.mlp.gate_proj`。
        tensors: 可以按 key 取张量的映射（safetensors 的惰性视图）。
        device: 目标设备。
        dtype: 逻辑 dtype，写进 `QuantizedTensor` 的元信息；实际计算精度由 kernel 决定。

    Returns:
        可直接参与 `F.linear` 的 QuantizedTensor。

    Raises:
        ValueError: 布局里 `format` 不是 `asym_w4a8_int8`。

    Notes:
        `group_size` / `convrot_groupsize` 从 `comfy_quant` blob 读，不写死：
        文件自己声明了布局参数，猜错会产生静默的数值错误。
        K 由「分组 scale 的列数 × 组大小」反推，不需要额外元数据。
    """
    config = read_quant_blob(tensors[f"{prefix}.comfy_quant"])
    if config.get("format") != QUANT_FORMAT:
        raise ValueError(f"{prefix}: 期望 {QUANT_FORMAT}，实际 {config.get('format')}")

    qdata = tensors[f"{prefix}.weight"].to(device)
    s_rel = tensors[f"{prefix}.weight_s_rel"].to(device)
    s_channel = tensors[f"{prefix}.weight_s_channel"].to(device)
    codebook = tensors.get(f"{prefix}.weight_codebook")
    codebook = None if codebook is None else codebook.to(device)

    group_size = int(config.get("group_size", 16))
    n = int(s_channel.shape[0])
    k = int(s_rel.shape[1]) * group_size

    params = AsymW4A8Int8Layout.Params(
        scale=s_rel,
        s_channel=s_channel,
        correction=None,
        codebook=codebook,
        orig_dtype=dtype,
        orig_shape=(n, k),
        group_size=group_size,
        convrot_groupsize=int(config.get("convrot_groupsize", 256)),
    )
    return QuantizedTensor(qdata, "AsymW4A8Int8Layout", params)


class QuantLinear(torch.nn.Module):
    """用 comfy_kitchen 量化权重替身的 `nn.Linear`。

    Args:
        weight: `build_asym_w4a8` 产出的量化张量。
        bias: 可选偏置 `[out_features]`。

    Notes:
        直接持有 `QuantizedTensor` 走 `F.linear`，由 comfy_kitchen 注册的
        `__torch_dispatch__` 分派到融合 kernel；权重始终以 int8 常驻显存，
        不展开成 bf16（9B 展开是 18.4 GB，8 GB 卡放不下）。
    """

    def __init__(self, weight: QuantizedTensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: `[..., in_features]`，bf16。

        Returns:
            `[..., out_features]`，dtype 与 `x` 一致。
        """
        return F.linear(x, self.weight, self.bias)


class Int8Embedding(torch.nn.Module):
    """int8_tensorwise 量化的词表查表。

    Args:
        weight: int8 `[vocab, hidden]`。
        scale: f32 `[vocab, 1]`，每行的反量化系数。
        dtype: 输出 dtype。

    Notes:
        反量化成 bf16 要 2 GB，和 6.84 GB 的 9B 权重一起会超出 8 GB 显存，
        所以保留 int8，只对本次命中的行做一次反量化。
    """

    def __init__(
        self, weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.weight = weight
        self.scale = scale
        self.out_dtype = dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """按 id 查表。

        Args:
            ids: `[B, T]` int64。

        Returns:
            `[B, T, hidden]`，dtype 为构造时的 `dtype`。

        Notes:
            先转 float32 再乘 scale，避免 bf16 下小 scale 丢有效位。
        """
        rows = self.weight[ids].to(torch.float32)
        return (rows * self.scale[ids]).to(self.out_dtype)
