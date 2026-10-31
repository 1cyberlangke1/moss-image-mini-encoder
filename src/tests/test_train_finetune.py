"""微调循环的纯函数测试：全部用合成数据，不读仓库 `tmp/` 下任何东西喵。

覆盖三件事：
- `make_batch` 是否切掉 token 0 并按长度右 padding；
- 长度对不齐的样本是否被丢弃而不是悄悄错位；
- `distill_loss` 的余弦与 CKA 分项是否与 numpy 独立实现一致喵。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moss_small_encoder.train.finetune import (
    PAD_ID,
    FinetuneConfig,
    distill_loss,
    make_batch,
)


class FakeTokenizer:
    """把字符映射成 id 的假分词器，避免加载 12 MB 词表喵。

    输入: 任意字符串
    输出: 每字符一个 id，偏移 `offset` 防止与特殊 token 冲突
    预期: 与真分词器同样提供 `encode(text) -> list[int]` 接口喵
    """

    def __init__(self, offset: int = 10) -> None:
        self.offset = offset

    def encode(self, text: str) -> list[int]:
        return [self.offset + ord(ch) for ch in text]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """批量接口：逐条复用 `encode`，语义与真分词器一致喵。"""
        return [self.encode(text) for text in texts]


def _stats(d_in: int, d_out: int) -> dict[str, torch.Tensor]:
    """恒等标准化统计量：均值 0、标准差 1，方便手算对照喵。"""
    return {
        "mean_in": torch.zeros(d_in),
        "std_in": torch.ones(d_in),
        "mean_out": torch.zeros(d_out),
        "std_out": torch.ones(d_out),
    }


def _numpy_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Kornblith 2019 的 numpy 实现（n×n Gram 与双重 centering）喵。"""

    def centering(k: np.ndarray) -> np.ndarray:
        n = k.shape[0]
        h = np.eye(n) - np.ones((n, n)) / n
        return h @ k @ h

    def hsic(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(centering(a @ a.T) * centering(b @ b.T)))

    return hsic(x, y) / np.sqrt(hsic(x, x) * hsic(y, y))


def test_make_batch_drops_sink_and_pads() -> None:
    tok = FakeTokenizer()
    texts = ["abc", "de"]
    teachers = [np.zeros((3, 4), dtype=np.float16), np.zeros((2, 4), dtype=np.float16)]

    ids, mask, kept = make_batch(tok, texts, teachers)

    # "abc" 去掉首 token 剩 2 个，"de" 剩 1 个，最长 2 喵
    assert ids.shape == (2, 2)
    assert mask.shape == (2, 2)
    assert ids[0, 0].item() == tok.encode("abc")[1]
    assert mask[0].tolist() == [1, 1]
    assert mask[1].tolist() == [1, 0]
    assert ids[1, 1].item() == PAD_ID
    assert kept[0].shape == (2, 4)
    assert kept[1].shape == (1, 4)


def test_make_batch_skips_mismatched_length() -> None:
    tok = FakeTokenizer()
    # 特征长度写成 5，与 "abc" 的 3 个 token 对不上，必须被丢弃喵
    teachers = [np.zeros((5, 4), dtype=np.float16), np.zeros((2, 4), dtype=np.float16)]

    ids, mask, kept = make_batch(tok, ["abc", "de"], teachers)

    assert ids.shape[0] == 1
    assert mask.tolist() == [[1]]
    assert kept[0].shape == (1, 4)


def test_make_batch_raises_when_nothing_left() -> None:
    tok = FakeTokenizer()
    teachers = [np.zeros((9, 4), dtype=np.float16)]
    with pytest.raises(RuntimeError):
        make_batch(tok, ["abc"], teachers)


def test_distill_loss_matches_numpy_reference() -> None:
    torch.manual_seed(0)
    d_in, d_out, n = 8, 4, 6
    hidden = torch.randn(n, d_in)
    teacher = torch.randn(n, d_out)
    # 这条验的是 fp32 路径的精确性，所以要显式关掉 bf16 喵
    config = FinetuneConfig(cka_bf16=False)

    loss, parts = distill_loss(hidden, teacher, config)

    # cka_loss 返回 1−√CKA；官方实现刻意用 sqrt 避免 CKA 接近 1 时梯度消失喵
    want_cka = 1.0 - float(np.sqrt(_numpy_cka(hidden.numpy().astype(np.float64),
                                              teacher.numpy().astype(np.float64))))
    assert parts["cka"] == pytest.approx(want_cka, abs=1e-5)

    want_total = config.cka_weight * want_cka + config.collapse_weight * parts["collapse"]
    assert float(loss) == pytest.approx(want_total, abs=1e-5)


def test_distill_loss_default_uses_bf16_within_tolerance() -> None:
    """默认配置走 bf16，结果与 fp32 参考的差必须在 1e-3 内喵。

    输入: 同一批随机张量，各建一份 bf16 与 fp32 配置
    输出: 两条路径的 cka 分项
    预期: 差 < 1e-3（实测约 1e-5，这里留足余量）喵
    """
    torch.manual_seed(3)
    d_in, d_out, n = 1024, 768, 24
    hidden = torch.randn(n, d_in) * 2.0
    teacher = torch.randn(n, d_out) * 2.0

    _, parts_bf16 = distill_loss(hidden, teacher, FinetuneConfig(cka_bf16=True))
    _, parts_fp32 = distill_loss(hidden, teacher, FinetuneConfig(cka_bf16=False))

    assert abs(parts_bf16["cka"] - parts_fp32["cka"]) < 1e-3


def test_distill_loss_zero_cka_when_teacher_equals_student() -> None:
    torch.manual_seed(1)
    d_in, n = 6, 7
    hidden = torch.randn(n, d_in)

    _, parts = distill_loss(hidden, hidden.clone(), FinetuneConfig())

    # 学生与教师逐位相同，CKA 必为 1，分项应为 0 喵
    assert parts["cka"] == pytest.approx(0.0, abs=1e-5)


def test_distill_loss_rejects_token_mismatch() -> None:
    with pytest.raises(ValueError):
        distill_loss(torch.randn(5, 4), torch.randn(6, 3), FinetuneConfig())
