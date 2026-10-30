"""线性投影：标准化空间里的 Ridge 闭式解。

方法照搬参考项目（ComfyUI-ClipProj 的 fit.py），只把维度换成 MOSS 这一路：

    cond = ((h - mean_in) / std_in) @ W * std_out + mean_out

几个容易做错的地方，这里都固定住：

- 正规方程按 **相关矩阵** 归一化再解，否则 λ 的量级没有意义；
- 精确解在 float64 里算，float32 解 1024×4096 会有可见误差；
- 第一个 token 是 attention sink（方向恒定、范数极大），必须排除拟合，
  由调用方在推理时用实测向量替换；
- 累加是流式的，显存/内存占用与语料规模无关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RidgeSolution:
    """一组可直接用于推理的线性投影参数。

    Attributes:
        W: `[d_in, d_out]` float32，在标准化空间里的映射矩阵。
        mean_in: `[d_in]` float32。
        std_in: `[d_in]` float32，已截断到最小 1e-12。
        mean_out: `[d_out]` float32。
        std_out: `[d_out]` float32，已截断到最小 1e-12。
        lam: 选中的正则强度。
        n_tokens: 参与拟合的 token 数。
    """

    W: np.ndarray
    mean_in: np.ndarray
    std_in: np.ndarray
    mean_out: np.ndarray
    std_out: np.ndarray
    lam: float
    n_tokens: int

    def predict(self, x: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """对新特征做投影。

        Args:
            x: `[n, d_in]`，原始（未标准化）student 特征，任意浮点类型。
            chunk: 每次处理的 token 数，控制内存。

        Returns:
            `[n, d_out]` float32 预测。

        Notes:
            分块是为了在 8 GB 机器上处理几十万 token 时不炸内存。
        """
        out = np.empty((x.shape[0], self.W.shape[1]), dtype=np.float32)
        for start in range(0, x.shape[0], chunk):
            block = np.asarray(x[start : start + chunk], dtype=np.float32)
            z = (block - self.mean_in) / self.std_in
            out[start : start + chunk] = (z @ self.W) * self.std_out + self.mean_out
        return out


class RidgeAccumulator:
    """流式累加 XᵀX / XᵀY 与一阶二阶矩。

    Args:
        d_in: student 维度。
        d_out: teacher 维度。
        drop_first: 每个 prompt 丢弃前几个 token（attention sink）。

    Notes:
        累加器只保留 `d_in² + d_in·d_out` 个 float64，1024/4096 时约 42 MB，
        与语料规模无关。
    """

    def __init__(self, d_in: int, d_out: int, drop_first: int = 1) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.drop_first = drop_first
        self.xx = np.zeros((d_in, d_in), dtype=np.float64)
        self.xy = np.zeros((d_in, d_out), dtype=np.float64)
        self.sx = np.zeros(d_in, dtype=np.float64)
        self.sy = np.zeros(d_out, dtype=np.float64)
        self.sx2 = np.zeros(d_in, dtype=np.float64)
        self.sy2 = np.zeros(d_out, dtype=np.float64)
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        """累加一个 prompt 的特征。

        Args:
            x: `[tokens, d_in]` student 特征。
            y: `[tokens, d_out]` teacher 特征（目标）。

        Notes:
            `drop_first` 是**按每次调用**生效的：调用方必须按 prompt 分块喂，
            一次喂整份语料只会丢掉第一行。长度不一致时取较短的那段：
            tokenizer 相同时不应发生，真发生了也要能继续跑。
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = min(x.shape[0], y.shape[0])
        start = self.drop_first if n > self.drop_first else n
        if start >= n:
            return
        x = x[start:n]
        y = y[start:n]
        # 矩阵乘用 float32（快一倍），结果累加进 float64（防止百万级 token 求和漂移）。
        # 参考项目是全程 float32 累加；我们多花一点内存换掉这个隐患。
        self.xx += (x.T @ x).astype(np.float64)
        self.xy += (x.T @ y).astype(np.float64)
        self.sx += x.sum(0, dtype=np.float64)
        self.sy += y.sum(0, dtype=np.float64)
        self.sx2 += (x * x).sum(0, dtype=np.float64)
        self.sy2 += (y * y).sum(0, dtype=np.float64)
        self.n += x.shape[0]

    def solve(self, lam: float) -> RidgeSolution:
        """在给定 λ 下解正规方程。

        Args:
            lam: 正则强度，作用在相关矩阵上。

        Returns:
            RidgeSolution。

        Raises:
            ValueError: 还没有累加任何 token。

        Notes:
            `cxx` 除以 n 变成相关矩阵（对角为 1），再加 `lam / n * I`，
            这样 λ 的量级与语料规模无关，1e1…1e6 的扫描才有可比性。
        """
        if self.n == 0:
            raise ValueError("没有累加到任何 token")
        n = self.n
        mx, my = self.sx / n, self.sy / n
        sdx = np.sqrt(np.maximum(self.sx2 / n - mx * mx, 1e-24))
        sdy = np.sqrt(np.maximum(self.sy2 / n - my * my, 1e-24))
        cxx = (self.xx - np.outer(self.sx, self.sx) / n) / (
            np.outer(sdx, sdx) * n
        )
        cxy = (self.xy - np.outer(self.sx, self.sy) / n) / (
            np.outer(sdx, sdy) * n
        )
        d = self.d_in
        w = np.linalg.solve(cxx + lam / n * np.eye(d, dtype=np.float64), cxy)
        return RidgeSolution(
            W=w.astype(np.float32),
            mean_in=mx.astype(np.float32),
            std_in=np.maximum(sdx, 1e-12).astype(np.float32),
            mean_out=my.astype(np.float32),
            std_out=np.maximum(sdy, 1e-12).astype(np.float32),
            lam=float(lam),
            n_tokens=n,
        )


def cosine(pred: np.ndarray, target: np.ndarray) -> float:
    """逐行余弦相似度的均值。

    Args:
        pred: `[n, d]` 预测。
        target: `[n, d]` 目标。

    Returns:
        `[-1, 1]` 之间的均值。两者都为零向量时该行记 0。

    Notes:
        余弦只反映方向，不反映幅度，所以必须和 `relative_l2` 一起看。
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dot = (pred * target).sum(1)
    norm = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    valid = norm > 0
    if not valid.any():
        return 0.0
    return float((dot[valid] / norm[valid]).mean())


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """相对 L2 误差。

    Args:
        pred: `[n, d]` 预测。
        target: `[n, d]` 目标。

    Returns:
        `||pred - target|| / ||target||`，整体而不是逐行平均。
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = np.linalg.norm(target)
    if denominator == 0:
        return float("inf")
    return float(np.linalg.norm(pred - target) / denominator)


def sweep(
    accumulator: RidgeAccumulator,
    lambdas: Iterable[float],
    evaluate,
) -> list[tuple[float, float, RidgeSolution]]:
    """在若干 λ 上求解并评估。

    Args:
        accumulator: 已累加完训练数据的累加器。
        lambdas: 待扫描的 λ。
        evaluate: 回调 `(solution) -> float`，返回该 λ 的验证分数（越大越好）。

    Returns:
        `[(λ, 分数, 解)]`，按 λ 升序。

    Notes:
        参考项目的结论是「余弦不能预测出图质量」，所以这里把每个 λ 的解都返回，
        调用方可以全部导出再逐张图测，而不是只留分数最高的那个。
    """
    results = []
    for lam in lambdas:
        solution = accumulator.solve(lam)
        results.append((float(lam), float(evaluate(solution)), solution))
    return results
