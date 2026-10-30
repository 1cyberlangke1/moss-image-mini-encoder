"""编码器共用逻辑的测试：RoPE 重建与线性注意力参数精度。

覆盖点：
- `restore_rope` 按 `rope_type` 选对初值函数，不总是用 default
- `restore_ssm_dtype` 把 `A_log` / `dt_bias` 拉回 fp32，其余参数不动
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from moss_small_encoder.encode.common import restore_rope, restore_ssm_dtype


def _rotary(rope_type: str) -> Qwen3_5TextRotaryEmbedding:
    """按给定 rope_type 造一个 RoPE 模块。

    Args:
        rope_type: `default` / `linear` / `dynamic` 等。

    Returns:
        已按配置初始化好 `inv_freq` 的模块。
    """
    params: dict = {
        "rope_type": rope_type,
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
    }
    if rope_type == "linear":
        params["factor"] = 2.0
    config = Qwen3_5TextConfig(head_dim=256, rope_parameters=params)
    return Qwen3_5TextRotaryEmbedding(config)


@pytest.mark.parametrize("rope_type", ["default", "linear"])
def test_restore_rope_honours_rope_type(rope_type: str) -> None:
    """重建出的频率必须与该 rope_type 的初值一致。

    输入: 一个被清零 inv_freq 的 RoPE 模块
    输出: 无（原地重建）
    预期: 与模块自己初始化时的 inv_freq 逐位相同。
        若实现写死 default 分支，linear 档在这里就会对不上喵
    """
    rot = _rotary(rope_type)
    expected = rot.inv_freq.clone()
    rot.inv_freq = torch.nn.Buffer(torch.zeros_like(expected), persistent=False)
    holder = torch.nn.Module()
    holder.rot = rot

    restore_rope(holder)

    assert torch.allclose(rot.inv_freq, expected, atol=0, rtol=0)
    assert torch.allclose(rot.original_inv_freq, expected, atol=0, rtol=0)


def test_restore_rope_rejects_unknown_type() -> None:
    """不认识的 rope_type 必须报错，不能悄悄退回 default。

    输入: rope_type 被改成 `warp`
    输出: 无（断言抛 ValueError）
    预期: 报错信息里点名该类型喵
    """
    rot = _rotary("default")
    rot.rope_type = "warp"
    holder = torch.nn.Module()
    holder.rot = rot

    with pytest.raises(ValueError, match="warp"):
        restore_rope(holder)


def test_restore_ssm_dtype_keeps_fp32() -> None:
    """`A_log` / `dt_bias` 被整体 cast 成 bf16 后要能拉回 fp32。

    输入: 两个参数都是 bf16 的假模块
    输出: 无（原地替换 Parameter）
    预期: 只有这两个变回 fp32，其余张量与 requires_grad 保持原样喵
    """
    model = torch.nn.Module()
    model.config = SimpleNamespace(mamba_ssm_dtype="float32")
    model.A_log = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
    model.dt_bias = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
    model.other = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    model.requires_grad_(False)

    restore_ssm_dtype(model)

    assert model.A_log.dtype == torch.float32
    assert model.dt_bias.dtype == torch.float32
    assert model.other.dtype == torch.bfloat16
    assert not model.A_log.requires_grad


def test_restore_ssm_dtype_defaults_to_float32() -> None:
    """config 没声明 mamba_ssm_dtype 时按 fp32 处理。

    输入: 没有该字段的假 config
    输出: 无
    预期: 参数仍被拉回 fp32，不抛异常喵
    """
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.A_log = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))

    restore_ssm_dtype(model)

    assert model.A_log.dtype == torch.float32
