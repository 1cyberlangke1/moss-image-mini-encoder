"""流式微调数据通路的回归测试喵。

覆盖两件事:
    1. `build_length_batches` 只改分组不改内容：每条样本恰好出现一次、
       不超 token 预算、同批长度接近（padding 浪费受控）；
    2. `PromptFeatureReader` 按任意顺序取下标都能读回正确的行。

预期: 全绿；任何一条挂了都说明训练会吃到串行/错位的教师特征喵。
"""

from __future__ import annotations

import numpy as np
import pytest

from moss_small_encoder.train.finetune import PromptFeatureReader, build_length_batches


def test_batches_cover_every_index_once() -> None:
    """批次合起来必须等于原下标集合，不多不少喵。

    输入: 200 条长度 5~500 的下标、预算 1024
    输出: 无
    预期: 展平后排序等于 0..199 喵
    """
    rng = np.random.default_rng(0)
    lengths = [int(v) for v in rng.integers(5, 500, size=200)]
    batches = build_length_batches(list(range(200)), lengths, 1024, rng)
    flat = sorted(i for batch in batches for i in batch)
    assert flat == list(range(200))


def test_batches_respect_padded_budget() -> None:
    """每批「条数 x 批内最长」不超过预算，这才是显存真正吃的东西喵。

    输入: 300 条随机长度、预算 4096
    输出: 无
    预期: 多条批的 B*T_max <= 4096；单条批允许自己就超预算喵
    """
    rng = np.random.default_rng(1)
    lengths = [int(v) for v in rng.integers(1, 900, size=300)]
    for batch in build_length_batches(list(range(300)), lengths, 4096, rng):
        if len(batch) > 1:
            assert max(lengths[i] for i in batch) * len(batch) <= 4096
        assert len(batch) <= 256


def test_batches_group_similar_lengths() -> None:
    """同批内长度差不能夸张，否则 pad 到最长那条会白烧显存喵。

    输入: 长度在 8~3000 之间极度分散的 400 条
    输出: 无
    预期: 真实 token 数不低于「最长 x 条数」的 60% 喵
    """
    rng = np.random.default_rng(2)
    lengths = [int(v) for v in rng.integers(8, 3000, size=400)]
    real = padded = 0
    for batch in build_length_batches(list(range(400)), lengths, 4096, rng):
        longest = max(lengths[i] for i in batch)
        real += sum(lengths[i] for i in batch)
        padded += longest * len(batch)
    assert real / padded > 0.6


def test_batches_cap_prompt_count() -> None:
    """超短句不能无限堆条数，否则 Python 侧与注意力核都会变慢喵。

    输入: 1000 条都只有 5 token、预算 1e6
    输出: 无
    预期: 每批不超过 256 条，且一条不丢喵
    """
    lengths = [5] * 1000
    batches = build_length_batches(list(range(1000)), lengths, 10**6, np.random.default_rng(3))
    assert all(len(b) <= 256 for b in batches)
    assert sum(len(b) for b in batches) == 1000


def test_batches_shuffle_between_calls() -> None:
    """同样的下标跑两轮，批内顺序应当不同（洗牌生效）喵。

    输入: 同一份下标与长度，两个不同随机源
    输出: 无
    预期: 至少有一批的顺序不一样喵
    """
    lengths = [100] * 40
    a = build_length_batches(list(range(40)), lengths, 400, np.random.default_rng(7))
    b = build_length_batches(list(range(40)), lengths, 400, np.random.default_rng(8))
    assert any(x != y for x, y in zip(a, b))


def test_feature_reader_roundtrip(tmp_path) -> None:
    """乱序读下标也要拿回对应那一条的行喵。

    输入: 3 条长度 2/3/4、维度 5 的假特征文件，按 [2,0,1] 读
    输出: 无
    预期: 每条的数值与写入时逐一相等喵
    """
    lengths = [2, 3, 4]
    d_out = 5
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    data = [np.arange(n * d_out, dtype=np.float16).reshape(n, d_out) + i * 100
            for i, n in enumerate(lengths)]
    (tmp_path / "teacher.raw").write_bytes(b"".join(d.tobytes() for d in data))

    reader = PromptFeatureReader(tmp_path, lengths, starts, d_out)
    try:
        got = reader.read([2, 0, 1])
    finally:
        reader.close()
    for row, want in zip(got, [data[2], data[0], data[1]]):
        np.testing.assert_array_equal(row, want)


def test_feature_reader_rejects_truncated_file(tmp_path) -> None:
    """文件被截断时必须报错，不能悄悄返回短数组喵。

    输入: 声称 3 行实际只写 2 行
    输出: 无
    预期: 抛 IOError 喵
    """
    lengths = [3]
    starts = np.array([0], dtype=np.int64)
    (tmp_path / "teacher.raw").write_bytes(np.zeros((2, 4), dtype=np.float16).tobytes())
    reader = PromptFeatureReader(tmp_path, lengths, starts, 4)
    try:
        with pytest.raises(IOError):
            reader.read([0])
    finally:
        reader.close()
