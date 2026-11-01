"""变长打包工具与打包分批的单元测试喵。

覆盖点：
- `packing_metadata` 的 `position_ids` 每段从 0 重开、`seq_idx` 逐段递增
- `packed_forward_kwargs` 的键齐全，`max_length` 取段内最长
- `build_packed_batches` 按真实 token 控预算且不丢不重
- `make_packed_batch` 拼平后偏移正确、教师侧同步切掉 token 0
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moss_small_encoder.packing import packed_forward_kwargs, packing_metadata
from moss_small_encoder.train.finetune import (
    PAD_ID,
    build_length_batches,
    build_packed_batches,
    build_tier_batches,
    default_pad_tiers,
    make_batch,
    make_packed_batch,
)


class _FakeTokenizer:
    """按文本返回预设 token 序列的分词器替身喵。"""

    def __init__(self, table: dict[str, list[int]]) -> None:
        self.table = table

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """返回每个文本对应的预设 token 列表喵。"""
        return [list(self.table[text]) for text in texts]


def test_position_ids_restart_per_segment() -> None:
    """每个段的位置从 0 重新开始。

    输入: cu_seqlens = [0, 3, 5]
    输出: position_ids 首行为 [0,1,2,0,1]，四条通道一致
    预期: 不重置会让第二段带上前一段的位置偏移喵
    """
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    position_ids, seq_idx = packing_metadata(cu)
    assert position_ids.shape == (4, 1, 5)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 0, 1]
    for row in range(1, 4):
        assert position_ids[row, 0].tolist() == [0, 1, 2, 0, 1]
    assert seq_idx.tolist() == [[0, 0, 0, 1, 1]]


def test_packed_kwargs_keys_and_max_length() -> None:
    """打包关键字齐全且 max_length 取段内最长。

    输入: cu_seqlens = [0, 4, 9]
    输出: 五个 varlen 键都在，max_length 为 5，attention_mask 是双 None 字典
    预期: 少一个键 flash 就走不到 varlen 分支喵
    """
    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    kwargs = packed_forward_kwargs(cu)
    for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx"):
        assert key in kwargs
    assert kwargs["max_length_q"] == 5
    assert kwargs["max_length_k"] == 5
    assert kwargs["attention_mask"] == {"full_attention": None, "linear_attention": None}


def test_build_packed_batches_respects_real_budget() -> None:
    """按真实 token 切批，预算不被补齐放大。

    输入: 20 条各 100 token，预算 500
    输出: 每批真实 token 不超过 500，且所有下标恰好出现一次
    预期: 每批 5 条，共 4 批喵
    """
    lengths = [100] * 20
    batches = build_packed_batches(list(range(20)), lengths, 500, np.random.default_rng(0))
    assert all(sum(lengths[i] for i in batch) <= 500 for batch in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(20))
    assert all(len(batch) == 5 for batch in batches)


def test_build_packed_batches_handles_mixed_lengths() -> None:
    """长短混合时也不超预算。

    输入: 长度 10 与 300 交替共 10 条，预算 620
    输出: 每批总和不超过 620
    预期: 打包不需要为最长样本补齐，短样本可以填满预算喵
    """
    lengths = [10, 300] * 5
    batches = build_packed_batches(list(range(10)), lengths, 620, np.random.default_rng(1))
    for batch in batches:
        assert sum(lengths[i] for i in batch) <= 620
    assert sorted(i for batch in batches for i in batch) == list(range(10))


def test_make_packed_batch_offsets_and_teacher_trim() -> None:
    """拼平后的偏移正确，教师侧同步切掉 token 0。

    输入: 两条长度 4 与 3 的样本
    输出: flat 是 1..5 共 5 个 token，cu_seqlens 为 [0, 3, 5]
    预期: 两侧都去掉 token 0，逐 token 仍然对齐喵
    """
    tokenizer = _FakeTokenizer({"a": [100, 1, 2, 3], "b": [200, 4, 5]})
    teachers = [
        np.arange(4 * 2, dtype=np.float16).reshape(4, 2),
        np.arange(3 * 2, dtype=np.float16).reshape(3, 2) + 100,
    ]
    flat_ids, cu_seqlens, kept = make_packed_batch(tokenizer, ["a", "b"], teachers)
    assert flat_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert cu_seqlens.tolist() == [0, 3, 5]
    assert [t.shape[0] for t in kept] == [3, 2]


def test_length_batches_respect_aspect_and_budget() -> None:
    """补齐批要同时守住长度比与补齐后预算喵。

    输入: 长度从 1 到 900 混排的下标
    输出: `build_length_batches` 的批次
    预期: 每批最长/最短不超过 3.0，每批「条数乘最长」不超过预算，
          且所有下标恰好出现一次喵
    """
    lengths = [1, 2, 3, 8, 20, 21, 60, 200, 900] * 6
    indices = list(range(len(lengths)))
    batches = build_length_batches(
        indices, lengths, 2048, np.random.default_rng(0), max_aspect=3.0
    )

    for batch in batches:
        lo = min(lengths[i] for i in batch)
        hi = max(lengths[i] for i in batch)
        assert hi <= lo * 3.0
        assert len(batch) * hi <= 2048

    assert sorted(x for batch in batches for x in batch) == indices


def test_default_pad_tiers_geometric() -> None:
    """默认档位要升序、盖住预算、相邻比值约 1.5 喵。

    输入: 预算 2048
    输出: `default_pad_tiers` 的档位表
    预期: 首档 16、末档等于预算、严格升序
    """
    tiers = default_pad_tiers(2048)
    assert tiers[0] == 16
    assert tiers[-1] == 2048
    assert all(a < b for a, b in zip(tiers, tiers[1:]))


def test_tier_batches_full_scale_invariants() -> None:
    """真实规模的不变量：批数量级、恰好覆盖、预算与档位一致喵。

    输入: 57.4 万条真实长度分布（档位 16..2048、预算 2048）
    输出: `build_tier_batches` 的批次
    预期: 批数在 1 万到 3 万之间（溢出 bug 会让它暴涨到 57 万）；
          所有下标恰好出现一次；每批条数乘批内最长不超过预算
          （独占批除外）；批形状种类不超过档位数加溢出批数喵
    """
    rng = np.random.default_rng(20261027)
    # 近似真实语料的长度分布：大量 8~40 token 短句 + 少量长文喵
    lengths = rng.lognormal(4.0, 1.1, size=574_087).astype(int).clip(3, 3200).tolist()
    indices = list(range(len(lengths)))
    tiers = [16, 24, 36, 54, 81, 121, 181, 271, 406, 609, 913, 1369, 2048]

    batches = build_tier_batches(indices, lengths, 2048, rng, tiers)

    # 溢出 bug 的特征是 57 万批（每批 1 条）；正常切批应低于其 1/5 喵
    assert len(batches) < 100_000, f"批数 {len(batches)} 呈溢出 bug 特征喵"
    # 碎批（<=2 条）只允许是档位尾巴：不超过两成喵
    tiny = sum(1 for b in batches if len(b) <= 2)
    assert tiny <= len(batches) * 0.2, f"碎批过多: {tiny}/{len(batches)} 喵"
    flat = sorted(i for b in batches for i in b)
    assert flat == indices
    for b in batches:
        longest = max(lengths[i] for i in b)
        if longest > tiers[-1]:
            assert len(b) == 1
        else:
            assert len(b) * longest <= 2048
            assert len(b) <= 256


def test_tier_batches_fixed_shapes_and_coverage() -> None:
    """档位切批要给出少数几种固定形状喵。

    输入: 长度从 1 到 900 混排的下标、档位 [32,128,512,1024]、预算 2048
    输出: `build_tier_batches` 的批次
    预期: 每批内所有样本都能进同一个档位；批形状（条数 x 档长）种类
          不超过档位数加超长独占批；所有下标恰好出现一次喵
    """
    lengths = [1, 2, 3, 8, 20, 21, 60, 200, 900] * 6
    indices = list(range(len(lengths)))
    tiers = [32, 128, 512, 1024]
    batches = build_tier_batches(
        indices, lengths, 2048, np.random.default_rng(0), tiers
    )

    shapes: set[tuple[int, int]] = set()
    for batch in batches:
        longest = max(lengths[i] for i in batch)
        if longest > tiers[-1]:
            assert len(batch) == 1
            shapes.add((1, longest))
            continue
        tier = min(t for t in tiers if t >= longest)
        assert len(batch) * tier <= 2048
        assert len(batch) <= 256
        shapes.add((len(batch), tier))

    assert len(shapes) <= len(tiers) + 2
    assert sorted(x for batch in batches for x in batch) == indices


def test_make_batch_pads_to_tier() -> None:
    """make_batch 给 pad_to 时要把 T 固定到档位喵。

    输入: 两条短文本、教师特征与 id 对齐、pad_to=128
    输出: `make_batch` 的批
    预期: T 恒为 128，mask 只盖有效位置，超出批内最长的部分是 PAD_ID 喵
    """

    class _Tok:
        @staticmethod
        def encode_batch(texts):
            return [[5, 6, 7], [8, 9]]

    teachers = [np.zeros((3, 4), dtype=np.float16), np.zeros((2, 4), dtype=np.float16)]
    input_ids, mask, kept = make_batch(_Tok(), ["a", "b"], teachers, pad_to=128)
    assert input_ids.shape == (2, 128)
    assert mask.shape == (2, 128)
    assert int(mask.sum()) == 3  # 第一条 2 个有效 + 第二条 1 个有效（去掉 token 0）
    assert int(input_ids[0, 3]) == PAD_ID
    assert len(kept) == 2


def test_make_packed_batch_pads_to_fixed_length() -> None:
    """补齐到固定总长时，末尾追加一个只含 padding 的假段喵。

    输入: 两条长度 4 与 3 的样本，`pad_to=10`（真实 5 个 token）
    输出: flat 长度 10，cu_seqlens 末尾多一段，teacher 仍是 5 行
    预期: 假段让每一步的激活形状一致，且不进入损失喵
    """
    tokenizer = _FakeTokenizer({"a": [100, 1, 2, 3], "b": [200, 4, 5]})
    teachers = [
        np.zeros((4, 2), dtype=np.float16),
        np.zeros((3, 2), dtype=np.float16),
    ]
    flat_ids, cu_seqlens, kept = make_packed_batch(
        tokenizer, ["a", "b"], teachers, pad_to=10
    )
    assert flat_ids.shape == (1, 10)
    assert cu_seqlens.tolist() == [0, 3, 5, 10]
    assert flat_ids[0, 5:].tolist() == [PAD_ID] * 5
    assert sum(int(t.shape[0]) for t in kept) == 5


def test_make_packed_batch_ignores_smaller_pad_to() -> None:
    """`pad_to` 小于真实长度时不做任何裁剪喵。

    输入: 真实 5 个 token，`pad_to=3`
    输出: 长度仍是 5，cu_seqlens 仍是 [0, 3, 5]
    预期: 补齐只增不减，绝不静默丢掉真实 token喵
    """
    tokenizer = _FakeTokenizer({"a": [100, 1, 2, 3], "b": [200, 4, 5]})
    teachers = [
        np.zeros((4, 2), dtype=np.float16),
        np.zeros((3, 2), dtype=np.float16),
    ]
    flat_ids, cu_seqlens, _ = make_packed_batch(
        tokenizer, ["a", "b"], teachers, pad_to=3
    )
    assert flat_ids.shape == (1, 5)
    assert cu_seqlens.tolist() == [0, 3, 5]


def test_make_packed_batch_rejects_mismatched_tokens() -> None:
    """token 数与特征行数对不上时整批报错。

    输入: 分词得 4 个 token，特征是 5 行
    输出: RuntimeError
    预期: 宁可整批丢掉也不静默错位喵
    """
    tokenizer = _FakeTokenizer({"a": [1, 2, 3, 4]})
    teachers = [np.zeros((5, 2), dtype=np.float16)]
    with pytest.raises(RuntimeError):
        make_packed_batch(tokenizer, ["a"], teachers)
