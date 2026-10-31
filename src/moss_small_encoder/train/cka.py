"""CKA 与塌缩正则：主干微调时用来对齐「token 之间的相对结构」。

为什么不直接用余弦：教师 4096 维、学生 1024 维，逐位余弦要求维度相同，
必须先经过一个线性头；而线性头一旦训歪，学生会被拉着去迁就那个头，
而不是真正学到教师的表示。CKA 只看两个空间里「哪些 token 互相接近」，
对旋转、缩放、维度都不敏感，所以先用它把结构对齐，再用线性头钉住基空间喵。

公式取自 ICLR 2025《Improving LM Distillation through Hidden State Matching》
的官方实现（Sayan21/ICLR25-CKA, MIT）：不构造 n×n 的 Gram，而用恒等式
`‖HtᵀHs‖_F`，交叉项最大只有 dS×dT，8192 token 也才几十 MB喵。

参考：arXiv 2605.30524 给出塌缩正则 `L_cos = 1/(m(m-1)) Σ_{i≠j} cos²(h̃i, h̃j)`，
h̃ 为去均值后的隐状态；我们把它当监测项，必要时才加进损失。
"""

from __future__ import annotations

import torch


def _norm_ratio(s: torch.Tensor, t: torch.Tensor, eps: float) -> torch.Tensor:
    """`‖tᵀs‖_F / sqrt(‖sᵀs‖_F·‖tᵀt‖_F)`，即 sqrt(CKA) 喵。

    输入: s [N,dS]、t [N,dT]，已去均值
    输出: 标量，1 表示结构完全一致
    预期: 交叉项只有 dS×dT，不构造 N×N 的 Gram
    """
    cross = torch.linalg.matrix_norm(s.t() @ t)
    den_s = torch.linalg.matrix_norm(s.t() @ s) + eps
    den_t = torch.linalg.matrix_norm(t.t() @ t) + eps
    return cross / torch.sqrt(den_s * den_t)


def linear_cka(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """线性 CKA 相似度，取值约 [0, 1]，1 表示结构完全一致。

    Args:
        student: `[N, dS]`，N 是 token 数，dS 是学生维度。
        teacher: `[N, dT]`，token 必须与 student 逐行对应。
        eps: 分母保护，防止常向量输入时出现 0/0。

    Returns:
        标量张量。内部用 float64 累加：4096 维交叉项在 fp32 下会有可见误差。

    Notes:
        两边都先按 token 维度去均值。去均值是在整批 token 上做的（跨 prompt），
        与官方实现一致；按 prompt 分别去均值的变体留给消融实验。

        定义上 CKA = HSIC(K,L) / sqrt(HSIC(K,K)·HSIC(L,L))，
        对线性核有 HSIC = ‖X̃ᵀỸ‖²_F，所以这里要把商的**平方**才等于 CKA 本身喵。
        官方实现里那个 `1 - ‖HtᵀHs‖/sqrt(...)` 是 `1 - sqrt(CKA)`，见 `cka_loss`。
    """
    if student.shape[0] != teacher.shape[0]:
        raise ValueError(f"token 数不一致：{student.shape[0]} vs {teacher.shape[0]}")
    s = student.reshape(-1, student.shape[-1]).to(torch.float64)
    t = teacher.reshape(-1, teacher.shape[-1]).to(torch.float64)
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)

    ratio = _norm_ratio(s, t, eps)
    return ratio * ratio


def cka_loss(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """`1 - sqrt(CKA)`，可以直接当损失反传。

    Args:
        student: `[N, dS]`，需要带梯度。
        teacher: `[N, dT]`，常量即可。
        eps: 同 `linear_cka`。

    Returns:
        标量张量，越小表示结构越接近。

    Notes:
        官方实现用的是 `1 - sqrt(CKA)`，即 `1 - ‖HtᵀHs‖/sqrt(‖HsᵀHs‖·‖HtᵀHt‖)`。
        用 sqrt 而不是 CKA 本身，是因为梯度在 CKA 接近 1 时更稳：
        `d(1-x²)/dx` 在 x=1 附近为 0，会卡住；`d(1-x)/dx` 恒为 -1喵。

        **梯度路径用 float32，不用 float64**：这个损失每步要做
        `‖sᵀs‖_F`（dS×dS 的 Gram）三次，代价是 `N·dS²`。
        在 RTX 4060 上 dS=6144、N≈3000 时是 237 GFLOP/次，
        Ada 消费卡的 fp64 只有 fp32 的 1/64，实测单步直接掉到
        42 秒、功耗 29 W（纯 fp64 空转），完全跑不动喵。
        fp32 累加 37.7M 个平方项的相对误差约 6e-4，落到损失上约 3e-4，
        相对 0.17 量级的损失可以忽略；需要精确读数时用 `linear_cka`（float64）喵。
    """
    if student.shape[0] != teacher.shape[0]:
        raise ValueError(f"token 数不一致：{student.shape[0]} vs {teacher.shape[0]}")
    s = student.reshape(-1, student.shape[-1]).to(torch.float32)
    t = teacher.reshape(-1, teacher.shape[-1]).to(torch.float32)
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)
    return 1.0 - _norm_ratio(s, t, eps)


def pairwise_cosine_penalty(hidden: torch.Tensor, max_tokens: int = 512) -> torch.Tensor:
    """塌缩正则：批内 token 两两余弦的平方均值。

    Args:
        hidden: `[N, d]`，通常是投影后的输出；N 超过 `max_tokens` 时等距抽样。
        max_tokens: 参与计算的最大 token 数。两两矩阵是 m×m，
            8192 全算会到 8192² = 268 MB，抽样后只留 512² = 1 MB。

    Returns:
        标量张量，全部 token 方向相同时趋近 1，互相正交时趋近 0。

    Notes:
        token 先逐行 L2 归一化再乘，等价于余弦；对角线（自己和自己，恒为 1）
        用 `fill_diagonal_` 抹掉，否则损失会被常数 1 顶住无法下降。
    """
    h = hidden.reshape(-1, hidden.shape[-1])
    if h.shape[0] > max_tokens:
        step = h.shape[0] // max_tokens
        h = h[::step][:max_tokens]
    h = torch.nn.functional.normalize(h, dim=-1)
    gram = h @ h.t()
    gram.fill_diagonal_(0.0)
    n = h.shape[0]
    if n < 2:
        return h.new_zeros(())
    return (gram ** 2).sum() / (n * (n - 1))


@torch.no_grad()
def token_pairwise_cosine(hidden: torch.Tensor, max_tokens: int = 512) -> float:
    """监测用：批内 token 两两余弦的均值（不是平方）。

    Args:
        hidden: `[N, d]`。
        max_tokens: 抽样上限。

    Returns:
        Python float。教师实测约 0.30~0.43，我们早期投影是 0.87，
        这个数往教师的方向降，说明塌缩在缓解。
    """
    h = hidden.reshape(-1, hidden.shape[-1])
    if h.shape[0] > max_tokens:
        step = h.shape[0] // max_tokens
        h = h[::step][:max_tokens]
    h = torch.nn.functional.normalize(h.float(), dim=-1)
    gram = h @ h.t()
    gram.fill_diagonal_(0.0)
    n = h.shape[0]
    if n < 2:
        return 0.0
    return float(gram.sum() / (n * (n - 1)))
