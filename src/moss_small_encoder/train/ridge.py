"""线性投影：标准化空间里的 Ridge 闭式解（GPU 流式累加）。

    cond = ((h - mean_in) / std_in) @ W * std_out + mean_out

数值约定：

- 正规方程按相关矩阵归一化再解，λ 的量级与语料规模无关；
- 求解在 float64 里做；
- 每条 prompt 的第一个 token 是 attention sink（范数极大），不参与拟合；
- 累加流式：磁盘顺序读 → 一块进 GPU → 算完统计量丢，任何时刻只有一块，
  与语料规模无关；无线程、无预取、无缓存。
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _device(device: str | None = None) -> torch.device:
    """解析 torch 设备：None 时有 GPU 用 GPU，否则 CPU。"""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def get_memory_info() -> tuple[float, float, float]:
    """返回 (进程物理内存RSS_GB, 系统可用内存_GB, 显卡已分配VRAM_GB)。"""
    import psutil
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024**3)
    sys_avail_gb = psutil.virtual_memory().available / (1024**3)
    vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return rss_gb, sys_avail_gb, vram_gb


def check_ram_limit(min_sys_available_gb: float = 2.0, max_process_ram_gb: float = 14.0) -> float:
    """真实硬件防护：基于系统全局可用内存熔断，防止系统卡顿与换页。

    参数说明：
    - min_sys_available_gb: 系统剩余物理内存底线（默认 2.0GB）。只要整机可用内存低于 2G，立即退出。
    - max_process_ram_gb: 进程私有物理内存异常上限（默认 14.0GB）。防范代码逻辑漏洞导致的单进程无限膨胀。
    """
    rss_gb, sys_avail_gb, _ = get_memory_info()
    if sys_avail_gb < min_sys_available_gb:
        raise MemoryError(
            f"【系统内存告急熔断】系统可用内存仅剩 {sys_avail_gb:.2f} GB < 警戒线 {min_sys_available_gb:.2f} GB！立即安全退出以保护系统流畅。"
        )
    if rss_gb > max_process_ram_gb:
        raise MemoryError(
            f"【进程内存异常熔断】进程物理内存异常膨胀到 {rss_gb:.2f} GB > 警戒线 {max_process_ram_gb:.2f} GB！立即退出。"
        )
    return rss_gb


def trim_working_set() -> None:
    """只回收进程工作集，不做 gc、不释放显存缓存。

    训练循环里每轮要过几十 GB 特征文件，页缓存会把进程工作集一路顶高
    （实测到 16GB，系统可用内存被打到 4GB 以下，随后 CUDA 分配失败）。
    这里只调 EmptyWorkingSet 把闲置页交还系统，代价远小于 trim_memory()，
    适合高频调用；不会被 release 掉的显存缓冲保持不动，避免分配器抖动。
    """
    import sys

    if sys.platform != "win32":
        return
    import ctypes

    try:
        # GetCurrentProcess 必须显式声明 64 位返回值，否则默认截断成 c_int 变成无效句柄
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_int
        psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except Exception:
        pass


def trim_memory() -> None:
    """主动通知垃圾回收器与系统归还闲置页，防止 RAM 与显存缓存持续堆积。"""
    import gc, sys
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if sys.platform == "win32":
        import ctypes
        try:
            # 必须显式声明 64 位指针类型，否则 GetCurrentProcess 默认截断为 32 位 c_int 导致无效句柄错误
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            psapi.EmptyWorkingSet.restype = ctypes.c_int
            psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
        except Exception:
            pass


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

    def predict(
        self, x: np.ndarray, chunk: int = 65536, device: str | None = None
    ) -> np.ndarray:
        """对新特征做投影（顺序分块，GPU 算）。

        Args:
            x: `[n, d_in]`，原始（未标准化）student 特征。可以是
                memmap：切片只触发对应页读入，天然流式。
            chunk: 每块的 token 数。
            device: torch 设备，None 时有 GPU 用 GPU。

        Returns:
            `[n, d_out]` float32 预测。
        """
        dev = _device(device)
        w = torch.from_numpy(self.W).to(dev)
        mean_in = torch.from_numpy(self.mean_in).to(dev)
        std_in = torch.from_numpy(self.std_in).to(dev)
        mean_out = torch.from_numpy(self.mean_out).to(dev)
        std_out = torch.from_numpy(self.std_out).to(dev)
        out = np.empty((x.shape[0], self.W.shape[1]), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, x.shape[0], chunk):
                # fp16 切片直传 GPU，块内转 fp32：CPU 侧零复制。
                block = (
                    torch.from_numpy(np.ascontiguousarray(x[start : start + chunk]))
                    .to(dev)
                    .float()
                )
                z = (block - mean_in) / std_in
                out[start : start + chunk] = (
                    ((z @ w) * std_out + mean_out).cpu().numpy()
                )
        return out

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
        chunk: int = 65536,
        device: str | None = None,
    ) -> tuple[float, float]:
        """流式算余弦与 relL2，不物化 `[n, d_out]` 预测矩阵。

        Args:
            x: `[n, d_in]` student 特征（可为 memmap）。
            y: `[n, d_out]` teacher 目标（可为 memmap）。
            chunk: 每块的 token 数。
            device: 同 `predict`。

        Returns:
            `(cosine, relative_l2)`，语义与下面的模块级函数一致。

        Notes:
            块内 fp32、块间 fp64 标量累加：内存峰值 = 单块，
            数值与全量 fp64 一致（求和顺序差异可忽略）。
        """
        dev = _device(device)
        w = torch.from_numpy(self.W).to(dev)
        mean_in = torch.from_numpy(self.mean_in).to(dev)
        std_in = torch.from_numpy(self.std_in).to(dev)
        mean_out = torch.from_numpy(self.mean_out).to(dev)
        std_out = torch.from_numpy(self.std_out).to(dev)
        dot = pn2 = tn2 = diff2 = tn_total = 0.0
        n_valid = 0
        with torch.inference_mode():
            for start in range(0, x.shape[0], chunk):
                stop = min(start + chunk, x.shape[0])
                xb = (
                    torch.from_numpy(np.ascontiguousarray(x[start:stop]))
                    .to(dev)
                    .float()
                )
                yb = (
                    torch.from_numpy(np.ascontiguousarray(y[start:stop]))
                    .to(dev)
                    .float()
                )
                z = (xb - mean_in) / std_in
                pred = (z @ w) * std_out + mean_out
                d_row = (pred * yb).sum(1)
                p_row = pred.square().sum(1)
                t_row = yb.square().sum(1)
                valid = (p_row * t_row) > 0
                dot += float(d_row[valid].double().sum())
                pn2 += float(p_row[valid].double().sum())
                tn2 += float(t_row[valid].double().sum())
                n_valid += int(valid.sum())
                diff2 += float((pred - yb).square().sum().double())
                tn_total += float(t_row.double().sum())
        cos = dot / (pn2**0.5 * tn2**0.5) if n_valid and pn2 > 0 else 0.0
        rel = (diff2**0.5) / (tn_total**0.5) if tn_total > 0 else float("inf")
        return float(cos), float(rel)

    def score_loader(
        self,
        loader,
        device: str | None = None,
    ) -> tuple[float, float]:
        """使用标准 PyTorch DataLoader 流式评估余弦与 relL2，零额外 RAM 占用。"""
        dev = _device(device)
        w = torch.from_numpy(self.W).to(dev)
        mean_in = torch.from_numpy(self.mean_in).to(dev)
        std_in = torch.from_numpy(self.std_in).to(dev)
        mean_out = torch.from_numpy(self.mean_out).to(dev)
        std_out = torch.from_numpy(self.std_out).to(dev)
        dot = pn2 = tn2 = diff2 = tn_total = 0.0
        n_valid = 0
        with torch.inference_mode():
            for bx, by in loader:
                if bx.shape[0] == 0:
                    continue
                xb = bx.to(dev, non_blocking=True).float()
                yb = by.to(dev, non_blocking=True).float()
                z = (xb - mean_in) / std_in
                pred = (z @ w) * std_out + mean_out
                d_row = (pred * yb).sum(1)
                p_row = pred.square().sum(1)
                t_row = yb.square().sum(1)
                valid = (p_row * t_row) > 0
                dot += float(d_row[valid].double().sum())
                pn2 += float(p_row[valid].double().sum())
                tn2 += float(t_row[valid].double().sum())
                n_valid += int(valid.sum())
                diff2 += float((pred - yb).square().sum().double())
                tn_total += float(t_row.double().sum())
        cos = dot / (pn2**0.5 * tn2**0.5) if n_valid and pn2 > 0 else 0.0
        rel = (diff2**0.5) / (tn_total**0.5) if tn_total > 0 else float("inf")
        return float(cos), float(rel)


class RidgeAccumulator:
    """流式累加 XᵀX / XᵀY 与一阶二阶矩（GPU）。

    Args:
        d_in: student 维度。
        d_out: teacher 维度。
        drop_first: 每个 prompt 丢弃前几个 token（attention sink）。
        device: torch 设备，None 时有 GPU 用 GPU。

    Notes:
        累加器本体 `d_in² + d_in·d_out` 个 float64（1024/4096 时 42 MB），
        与语料规模无关。调用方按磁盘顺序切一块喂一次 `add`，
        块大小只影响吞吐，不影响结果。
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        drop_first: int = 1,
        device: str | None = None,
    ) -> None:
        if drop_first < 0:
            raise ValueError(f"drop_first 必须 >= 0，得到 {drop_first}")
        self.d_in = d_in
        self.d_out = d_out
        self.drop_first = drop_first
        self.device = _device(device)
        # 累加器用 float64（防百万级求和漂移）；GEMM 用 fp32——
        # fp16 特征本身只有 10 位尾数，fp32 GEMM + fp64 累加足够。
        d = self.device
        self.xx = torch.zeros(d_in, d_in, dtype=torch.float64, device=d)
        self.xy = torch.zeros(d_in, d_out, dtype=torch.float64, device=d)
        self.sx = torch.zeros(d_in, dtype=torch.float64, device=d)
        self.sy = torch.zeros(d_out, dtype=torch.float64, device=d)
        self.sx2 = torch.zeros(d_in, dtype=torch.float64, device=d)
        self.sy2 = torch.zeros(d_out, dtype=torch.float64, device=d)
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray, lengths=None, clean: bool = False) -> None:
        """累加一块特征（顺序读入，GPU 算）。

        Args:
            x: `[tokens, d_in]` student 特征。
            y: `[tokens, d_out]` teacher 特征，与 x 逐行对齐。
            lengths: 该块内每个 prompt 的 token 数（None 表示单条 prompt）。
            clean: 若为 True，表示调用方已在读取层面剔除了 sink token，无需二次切片。
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x {x.shape[0]} token != y {y.shape[0]} token，必须逐行对齐"
            )
        if x.shape[0] == 0:
            return

        if clean or self.drop_first == 0:
            xt = torch.from_numpy(x).to(self.device).float()
            yt = torch.from_numpy(y).to(self.device).float()
        else:
            if lengths is None:
                lengths = [x.shape[0]]
            total = int(np.sum(lengths))
            if total != x.shape[0]:
                raise ValueError(f"lengths 之和 {total} != 行数 {x.shape[0]}")

            # 去掉每段开头的 sink 行
            keep = []
            offset = 0
            for length in lengths:
                length = int(length)
                start = offset + min(self.drop_first, length)
                if start < offset + length:
                    keep.append(np.arange(start, offset + length))
                offset += length
            if not keep:
                return
            rows = np.concatenate(keep) if len(keep) > 1 else keep[0]

            # 上传去 sink 后的行（fp16 → GPU → fp32），GEMM 整块一次
            xt = (
                torch.from_numpy(np.ascontiguousarray(x[rows]))
                .to(self.device)
                .float()
            )
            yt = (
                torch.from_numpy(np.ascontiguousarray(y[rows]))
                .to(self.device)
                .float()
            )

        self.xx += (xt.T @ xt).to(torch.float64)
        self.xy += (xt.T @ yt).to(torch.float64)
        # 统计量按块归约：块内 fp32 求和，块结果再进 fp64。
        # 直接写 .sum(0, dtype=torch.float64) 会把上亿个元素逐个转成 fp64 再相加，
        # 而消费级 GPU 的 fp64 只有 fp32 的 1/64（RTX 4060 约 230 GFLOPS），
        # 实测这几处归约占掉整轮累加的大头。分块后误差只看块内几千项，
        # 平方和非负无抵消、均值有界，对标准化用的 mean/std 足够。
        self.sx += self._chunked_sum(xt)
        self.sy += self._chunked_sum(yt)
        self.sx2 += self._chunked_sum_squares(xt)
        self.sy2 += self._chunked_sum_squares(yt)
        self.n += xt.shape[0]
        del xt, yt

    @staticmethod
    def _chunked_sum(x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """按块求列和，块结果累进 fp64。

        Args:
            x: `[tokens, dim]`，fp32。
            chunk: 每块行数，控制块内 fp32 求和的误差累积。

        Returns:
            `[dim]` float64。
        """
        total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        for start in range(0, x.shape[0], chunk):
            total += x[start : start + chunk].sum(0).to(torch.float64)
        return total

    @staticmethod
    def _chunked_sum_squares(x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """按块求列平方和，块结果累进 fp64。

        Args:
            x: `[tokens, dim]`，fp32。
            chunk: 每块行数；块内先物化平方再求和，块越小峰值显存越低。

        Returns:
            `[dim]` float64。
        """
        total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        for start in range(0, x.shape[0], chunk):
            total += x[start : start + chunk].square().sum(0).to(torch.float64)
        return total

    def solve(self, lam: float) -> RidgeSolution:
        """在给定 λ 下解正规方程。

        Args:
            lam: 正则强度，作用在相关矩阵上。

        Returns:
            RidgeSolution。

        Raises:
            ValueError: 还没有累加任何 token。

        Notes:
            `cxx` 归一化成相关矩阵、加 `lam / n * I` 再解，λ 的量级
            与语料规模无关，1e1…1e6 的扫描才有可比性。
        """
        if self.n == 0:
            raise ValueError("没有累加到任何 token")
        n = self.n
        # 全程 GPU：统计量在卡上，矩阵也在卡上解（torch.linalg.solve
        # 的 CUDA 路径走 cuSOLVER），不回 CPU。
        mx, my = self.sx / n, self.sy / n
        sdx = torch.sqrt(torch.clamp(self.sx2 / n - mx * mx, min=1e-24))
        sdy = torch.sqrt(torch.clamp(self.sy2 / n - my * my, min=1e-24))
        cxx = (self.xx - torch.outer(self.sx, self.sx) / n) / (
            torch.outer(sdx, sdx) * n
        )
        cxy = (self.xy - torch.outer(self.sx, self.sy) / n) / (
            torch.outer(sdx, sdy) * n
        )
        # 正规方程矩阵对称正定：用 Cholesky（potrf，n³/3）比通用 solve（2n³/3）省一半。
        # 主解放在 fp32（RTX 4060 的 fp32 是 fp64 的 64 倍），再用 fp64 残差做 1~2 次
        # 迭代精化，精度回到 fp64 水平；每次精化只是一次矩阵乘，代价可忽略。
        # 不用 torch.eye 造单位阵：6144² 的 fp64 单位阵要多占 300 MB 显存。
        normal = cxx.clone()
        normal.diagonal().add_(lam / n)
        a32 = normal.to(torch.float32)
        b32 = cxy.to(torch.float32)
        factor = None
        try:
            factor = torch.linalg.cholesky(a32)
            x64 = torch.cholesky_solve(b32, factor).to(torch.float64)
        except RuntimeError:
            # λ 极小时矩阵可能在数值上失去正定性，退回通用求解
            x64 = torch.linalg.solve(a32, b32).to(torch.float64)
        for _ in range(2):
            residual = (cxy - normal @ x64).to(torch.float32)
            if factor is not None:
                correction = torch.cholesky_solve(residual, factor)
            else:
                correction = torch.linalg.solve(a32, residual)
            x64 = x64 + correction.to(torch.float64)
        w = x64
        return RidgeSolution(
            W=w.float().cpu().numpy(),
            mean_in=mx.float().cpu().numpy(),
            std_in=torch.clamp(sdx, min=1e-12).float().cpu().numpy(),
            mean_out=my.float().cpu().numpy(),
            std_out=torch.clamp(sdy, min=1e-12).float().cpu().numpy(),
            lam=float(lam),
            n_tokens=n,
        )


# ---------------------------------------------------------------------------
# Ridge 结果缓存
#
# 累加（4.5M token 约 100 秒）与 λ 扫描（每个 λ 十几秒）只取决于「特征库 + 口径」，
# 跟之后的 MLP 实验无关，重复训练时没有理由重算。分两层缓存、各自独立失效：
#   ridge_stats_<key>.npz      累加器本体，只取决于 meta.json 与 drop_first；
#   ridge_solution_<key>.npz   选定 λ 的解，额外取决于 λ 网格与验证 token 上限。
# 任何一项变了键就变，旧文件自然不再被命中（不删除，避免误删他人实验）。
# ---------------------------------------------------------------------------

_RIDGE_CACHE_VERSION = 1


def ridge_stats_key(meta_bytes: bytes, drop_first: int) -> str:
    """累加器缓存的键。

    Args:
        meta_bytes: 特征目录下 `meta.json` 的原始字节。其中含 prompt_ids 与 lengths，
            换一批特征（重编码、改顺序）就必然不同。
        drop_first: 每条 prompt 丢弃的前导 token 数（attention sink）。

    Returns:
        16 位十六进制字符串，用作缓存文件名后缀。
    """
    digest = hashlib.sha1()
    digest.update(f"v{_RIDGE_CACHE_VERSION}|drop={drop_first}|".encode())
    digest.update(meta_bytes)
    return digest.hexdigest()[:16]


def ridge_solution_key(stats_key: str, lambdas: Iterable[float], eval_tokens: int) -> str:
    """解缓存的键。

    Args:
        stats_key: 上一层累加缓存的键。
        lambdas: λ 网格。换网格会选出另一个解，必须让旧缓存失效。
        eval_tokens: 验证集实际使用的 token 上限，同样会改变最优 λ。

    Returns:
        16 位十六进制字符串。
    """
    digest = hashlib.sha1()
    digest.update(f"{stats_key}|eval={eval_tokens}|lam=".encode())
    digest.update(",".join(f"{value:.6g}" for value in lambdas).encode())
    return digest.hexdigest()[:16]


def save_accumulator(path: Path, accumulator: "RidgeAccumulator") -> None:
    """把累加器统计量写盘。

    Args:
        path: 目标 `.npz` 路径。
        accumulator: 已累加过数据的累加器。

    Returns:
        无，直接写文件。d_in=6144 / d_out=4096 时约 503 MB（未压缩，压缩对随机浮点无效）。
    """
    np.savez(
        path,
        xx=accumulator.xx.detach().cpu().numpy(),
        xy=accumulator.xy.detach().cpu().numpy(),
        sx=accumulator.sx.detach().cpu().numpy(),
        sy=accumulator.sy.detach().cpu().numpy(),
        sx2=accumulator.sx2.detach().cpu().numpy(),
        sy2=accumulator.sy2.detach().cpu().numpy(),
        n=np.array(accumulator.n, dtype=np.int64),
    )


def load_accumulator(
    path: Path, drop_first: int, device: str | None = None
) -> "RidgeAccumulator | None":
    """读回累加器。

    Args:
        path: 缓存文件。
        drop_first: 与写盘时一致的口径。
        device: 目标 torch 设备，None 时有 GPU 用 GPU。

    Returns:
        `RidgeAccumulator`；文件不存在或读坏了返回 None。

    Notes:
        读坏了只降级成「重算」，不抛异常：缓存是加速手段，不该成为训练失败的原因。
    """
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            accumulator = RidgeAccumulator(
                int(data["xx"].shape[0]),
                int(data["xy"].shape[1]),
                drop_first=drop_first,
                device=device,
            )
            target = accumulator.device
            accumulator.xx = torch.from_numpy(data["xx"]).to(target)
            accumulator.xy = torch.from_numpy(data["xy"]).to(target)
            accumulator.sx = torch.from_numpy(data["sx"]).to(target)
            accumulator.sy = torch.from_numpy(data["sy"]).to(target)
            accumulator.sx2 = torch.from_numpy(data["sx2"]).to(target)
            accumulator.sy2 = torch.from_numpy(data["sy2"]).to(target)
            accumulator.n = int(data["n"])
    except (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None
    return accumulator


def save_solution(
    path: Path,
    solution: RidgeSolution,
    grid: Iterable[tuple[float, float, float]],
) -> None:
    """把选定的解连同 λ 扫描记录一起写盘。

    Args:
        path: 目标 `.npz` 路径。
        solution: 选中的 RidgeSolution。
        grid: `(lam, cos, relL2)` 三元组序列，供下次直接回放日志。

    Returns:
        无。d_in=6144 / d_out=4096 时约 100 MB。
    """
    rows = list(grid)
    np.savez(
        path,
        W=solution.W,
        mean_in=solution.mean_in,
        std_in=solution.std_in,
        mean_out=solution.mean_out,
        std_out=solution.std_out,
        lam=np.array(solution.lam, dtype=np.float64),
        n=np.array(solution.n_tokens, dtype=np.int64),
        grid_lam=np.array([row[0] for row in rows], dtype=np.float64),
        grid_cos=np.array([row[1] for row in rows], dtype=np.float64),
        grid_rel=np.array([row[2] for row in rows], dtype=np.float64),
    )


def load_solution(
    path: Path,
) -> "tuple[RidgeSolution, list[tuple[float, float, float]]] | None":
    """读回解与 λ 扫描记录。

    Args:
        path: 缓存文件。

    Returns:
        `(solution, grid)`；文件不存在或读坏了返回 None。
    """
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            solution = RidgeSolution(
                W=data["W"],
                mean_in=data["mean_in"],
                std_in=data["std_in"],
                mean_out=data["mean_out"],
                std_out=data["std_out"],
                lam=float(data["lam"]),
                n_tokens=int(data["n"]),
            )
            grid = [
                (float(lam), float(cos), float(rel))
                for lam, cos, rel in zip(data["grid_lam"], data["grid_cos"], data["grid_rel"])
            ]
    except (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None
    return solution, grid


def cosine(pred: np.ndarray, target: np.ndarray) -> float:
    """逐行余弦相似度的均值（小数组用；大数组用 `RidgeSolution.score`）。

    Args:
        pred: `[n, d]` 预测。
        target: `[n, d]` 目标。

    Returns:
        `[-1, 1]` 之间的均值。两者都为零向量时该行记 0。
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
    """相对 L2 误差（小数组用；大数组用 `RidgeSolution.score`）。

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
