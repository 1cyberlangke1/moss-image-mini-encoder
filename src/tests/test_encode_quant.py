"""int8_tensorwise 词表查表的反量化测试。

覆盖点：
- `convrot=True` 时必须做逆旋转，否则拿到的是旋转后基底（范数相同、方向全错）
- `convrot=False` 时退化为 `int8 * scale` 的逐行反量化
- 多维 id 的形状还原
"""

from __future__ import annotations

import torch
from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight

from moss_small_encoder.encode.quant import Int8Embedding


def _make_rotated_table(rows: int = 8, dim: int = 512, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """造一张真实的 ConvRot int8 词表。

    Args:
        rows: 词表行数。
        dim: 每行维度，必须是 convrot 分组宽度的整数倍。
        seed: 随机种子，保证可复现。

    Returns:
        `(原始 fp32 权重, int8 权重, scale [rows,1])`。
    """
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, dim, generator=generator)
    q, scale = quantize_int8_convrot_weight(weight, 256)
    return weight, q, scale.reshape(rows, 1)


def test_convrot_true_recovers_original_basis() -> None:
    """convrot=True 时查表结果应与原始权重同向。

    输入: 由 `quantize_int8_convrot_weight` 生成的 int8 表（已旋转）
    输出: 逐行余弦 > 0.999，相对 L2 小于 2%（int8 量化本身的误差）
    预期: 不做逆旋转时余弦会掉到 0.1 以下，所以这条能抓住漏旋转的回归
    """
    weight, q, scale = _make_rotated_table()
    table = Int8Embedding(q, scale, dtype=torch.float32, convrot=True, convrot_groupsize=256)

    out = table(torch.arange(weight.shape[0]))
    cosine = torch.nn.functional.cosine_similarity(out, weight, dim=-1)
    relative = (out - weight).norm() / weight.norm()

    assert float(cosine.min()) > 0.999
    assert float(relative) < 0.02


def test_convrot_false_is_plain_rowwise_dequant() -> None:
    """convrot=False 时就是逐行 int8 * scale。

    输入: 同一张已旋转的表，但按未旋转处理
    输出: 与 `q.float() * scale` 完全相等
    预期: 这条锁定"没有旋转时不要多算"，避免修复后对所有 int8 表都强行逆旋转
    """
    _, q, scale = _make_rotated_table()
    table = Int8Embedding(q, scale, dtype=torch.float32, convrot=False)

    ids = torch.arange(q.shape[0])
    expected = q[ids].float() * scale[ids]
    assert torch.equal(table(ids), expected)


def test_multidim_ids_restore_shape() -> None:
    """多维 id 要还原成 `[*ids.shape, hidden]`。

    输入: ids 形状 `[2, 3]`
    输出: 形状 `[2, 3, 512]`，且与按 1D 查表再 reshape 的结果一致
    预期: 展平/还原过程不串行
    """
    _, q, scale = _make_rotated_table()
    table = Int8Embedding(q, scale, dtype=torch.float32, convrot=True, convrot_groupsize=256)

    ids = torch.arange(6).reshape(2, 3)
    assert table(ids).shape == (2, 3, q.shape[1])
    assert torch.equal(table(ids).reshape(-1, q.shape[1]), table(torch.arange(6)))
