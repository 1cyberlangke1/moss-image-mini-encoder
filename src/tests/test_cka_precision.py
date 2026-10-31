"""CKA 数值精度的单元测试，全部在 CPU 上跑喵。"""

from __future__ import annotations

import torch

from moss_small_encoder.train.cka import cka_loss, linear_cka


def test_default_dtype_is_fp32() -> None:
    """不传 dtype 时必须等价于显式 fp32喵。

    输入: 同一批随机张量
    输出: 两次 `cka_loss` 的标量
    预期: 逐位相同喵
    """
    torch.manual_seed(0)
    student = torch.randn(32, 128)
    teacher = torch.randn(32, 96)

    default = cka_loss(student, teacher)
    explicit = cka_loss(student, teacher, dtype=torch.float32)

    assert torch.equal(default, explicit)


def test_bf16_close_to_fp32() -> None:
    """bf16 路径与 fp32 路径的 CKA 差必须小于 1e-3喵。

    输入: 学生 [32,1024]、教师 [32,768]，量级贴近真实隐状态
    输出: 两条路径的 CKA 值
    预期: 绝对差 < 1e-3；这是 bf16 尾数 8 位的理论上界喵
    """
    torch.manual_seed(1)
    student = torch.randn(32, 1024) * 2.0
    teacher = torch.randn(32, 768) * 2.0

    fp32 = cka_loss(student, teacher, dtype=torch.float32)
    bf16 = cka_loss(student, teacher, dtype=torch.bfloat16)

    assert abs(float(fp32) - float(bf16)) < 1e-3


def test_bf16_keeps_gradient_path() -> None:
    """bf16 路径必须能反传，且梯度有限、非全零喵。

    输入: 需要梯度的学生张量
    输出: 损失的梯度
    预期: 梯度形状一致、全为有限值、至少有一个非零元素喵
    """
    torch.manual_seed(2)
    student = torch.randn(16, 256, requires_grad=True)
    teacher = torch.randn(16, 192)

    loss = cka_loss(student, teacher, dtype=torch.bfloat16)
    loss.backward()

    assert student.grad is not None
    assert student.grad.shape == student.shape
    assert torch.isfinite(student.grad).all()
    assert float(student.grad.abs().sum()) > 0.0


def test_identical_inputs_score_one() -> None:
    """教师与学生逐位相同时 CKA 必须是 1，损失为 0喵。"""
    torch.manual_seed(3)
    hidden = torch.randn(24, 128)

    loss = cka_loss(hidden, hidden.clone(), dtype=torch.bfloat16)

    assert abs(float(loss)) < 1e-5


def test_linear_cka_keeps_fp64_path() -> None:
    """`linear_cka` 仍走 fp64，不能因为 Gram 改动被降成 fp32喵。

    输入: 同一批 fp32 张量
    输出: `linear_cka` 的结果
    预期: 与 fp64 手算平方后的 CKA 相差 < 1e-5喵
    """
    torch.manual_seed(4)
    student = torch.randn(20, 64)
    teacher = torch.randn(20, 48)

    got = float(linear_cka(student, teacher))

    s = student.to(torch.float64)
    t = teacher.to(torch.float64)
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(s.t() @ t)
    den = torch.sqrt(
        torch.linalg.matrix_norm(s.t() @ s) * torch.linalg.matrix_norm(t.t() @ t)
    )
    want = float((cross / den) ** 2)

    assert abs(got - want) < 1e-5
