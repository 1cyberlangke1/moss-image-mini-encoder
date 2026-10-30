"""CKA 实现与 numpy 经典实现对拍，防止公式抄错喵。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moss_small_encoder.train.cka import (
    cka_loss,
    linear_cka,
    pairwise_cosine_penalty,
    token_pairwise_cosine,
)


def _numpy_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Kornblith 2019 的 numpy 实现（对照 oracle）。

    输入: x [N,d1]、y [N,d2]
    输出: CKA 标量
    预期: 与 torch 版逐位接近（1e-9 量级）
    """
    def centering(k: np.ndarray) -> np.ndarray:
        n = k.shape[0]
        h = np.eye(n) - np.ones((n, n)) / n
        return h @ k @ h

    def hsic(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(centering(a @ a.T) * centering(b @ b.T)))

    return hsic(x, y) / np.sqrt(hsic(x, x) * hsic(y, y))


def test_matches_numpy_oracle_on_random_matrices() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 7))
    y = rng.normal(size=(64, 11))

    got = float(linear_cka(torch.from_numpy(x), torch.from_numpy(y)))
    want = _numpy_linear_cka(x, y)

    assert got == pytest.approx(want, rel=1e-9, abs=1e-9)


def test_self_similarity_is_one() -> None:
    x = torch.randn(48, 5, dtype=torch.float64)
    assert float(linear_cka(x, x)) == pytest.approx(1.0, rel=1e-9)


def test_invariant_to_rotation_and_scale_of_one_side() -> None:
    x = torch.randn(40, 6, dtype=torch.float64)
    y = torch.randn(40, 9, dtype=torch.float64)
    base = float(linear_cka(x, y))

    # 对学生的特征做正交旋转 + 整体缩放，CKA 不该变喵
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    rotated = 3.5 * (x @ q)
    assert float(linear_cka(rotated, y)) == pytest.approx(base, rel=1e-9)


def test_independent_noise_has_low_cka() -> None:
    x = torch.randn(128, 8, dtype=torch.float64)
    y = torch.randn(128, 8, dtype=torch.float64)
    assert float(linear_cka(x, y)) < 0.4


def test_cka_loss_is_zero_at_perfect_alignment() -> None:
    x = torch.randn(32, 4)
    assert float(cka_loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_cka_loss_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        linear_cka(torch.randn(10, 3), torch.randn(11, 5))


def test_pairwise_cosine_penalty_extremes() -> None:
    """同向时接近 1（对角线已抹掉），两两正交时接近 0 喵。"""
    same = torch.randn(1, 16).repeat(32, 1)
    assert float(pairwise_cosine_penalty(same)) == pytest.approx(1.0, abs=1e-5)

    eye = torch.eye(32)
    assert float(pairwise_cosine_penalty(eye)) == pytest.approx(0.0, abs=1e-6)


def test_pairwise_cosine_penalty_subsamples() -> None:
    h = torch.randn(2000, 16)
    value = float(pairwise_cosine_penalty(h, max_tokens=128))
    assert np.isfinite(value) and 0.0 <= value <= 1.0


def test_token_pairwise_cosine_matches_manual_mean() -> None:
    h = torch.randn(16, 8)
    manual = token_pairwise_cosine(h)
    unit = torch.nn.functional.normalize(h, dim=-1)
    gram = unit @ unit.t()
    gram.fill_diagonal_(0.0)
    assert manual == pytest.approx(float(gram.sum() / (16 * 15)), rel=1e-6)
