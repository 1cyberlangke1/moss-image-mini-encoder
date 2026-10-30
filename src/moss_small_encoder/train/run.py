"""一键训练：Ridge λ 扫描 → 残差网络 → 导出。

动态适配显卡与内存控制：
- 自动根据显卡显存自适应 batch 与流式切块大小（或通过 CLI --batch / --chunk-tokens 显式指定）：
  * 8GB 显卡（如 4060）：自动设 batch=32768，吃满 ~6.5GB VRAM；
  * 16GB 显卡（如 4080/T4x2）：自动设 batch=65536，吃满 ~13GB VRAM；
  * 24GB+ 显卡（如 3090/4090/A100）：自动设 batch=98304。
- 内存（RAM）：全程复用同一套缓冲数组（仅约 670MB），绝对零额外全量申请。
- 安全熔断：实时通过 psutil 监测，若进程 RAM > 5.5 GB 或系统可用 < 4.0 GB 立即退出保护系统。

用法：
    python -m moss_small_encoder.train.run --features dataset/features/v1 --smoke
    python -m moss_small_encoder.train.run --features dataset/features/v1
    python -m moss_small_encoder.train.run --features dataset/features/v1 --batch 32768
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .mlp import MLPConfig, save, save_linear, train
from .ridge import (
    RidgeAccumulator,
    check_ram_limit,
    get_memory_info,
    load_accumulator,
    load_solution,
    ridge_solution_key,
    ridge_stats_key,
    save_accumulator,
    save_solution,
    trim_memory,
)
from .store import StreamReader

# 网格上界必须远超预期最优：若最优落在边界，说明真正的最优在网格外（欠拟合）。
# 参考项目文档专门警告过这一点，而我们原先的上界 1e6 恰好被选中。
LAMBDAS = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)


def auto_detect_hardware(
    device: str,
    user_batch: int | None,
    user_chunk: int | None,
    d_in: int = 1024,
    d_out: int = 4096,
) -> tuple[int, int]:
    """根据显存与特征宽度自适应 batch 和流式读取 chunk。

    Args:
        device: `cuda` 或 `cpu`。
        user_batch: 显式指定的 batch，None 表示自动。
        user_chunk: 显式指定的 chunk，None 表示自动。
        d_in: 输入维度（多 tap 拼接后是 1024 x 层数）。
        d_out: 目标维度。

    Returns:
        `(batch, chunk)`，单位 token。

    Notes:
        只看显存挑 batch 是不够的：训练端还有两块**按 (chunk + batch) 分配的 fp32
        固定缓冲**（x 与 y），以及随 batch 线性增长的激活值。它们的开销正比于
        `batch * (d_in + 1024)`，所以 d_in 从 1024 涨到 6144 时同样的批次会顶穿 8 GB
        （实测 d_in=6144、batch=32768 时共享显存吃到 6.1 GB，单步从 0.2 秒掉到 24 秒）。
        这里再乘一个与 d_in 成反比的系数做上限。
    """
    if device == "cuda" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        vram_gb = 0.0

    if user_batch is not None:
        batch = user_batch
    elif vram_gb >= 22:
        batch = 98_304
    elif vram_gb >= 14:
        batch = 65_536
    elif vram_gb >= 7:
        batch = 32_768
    else:
        batch = 16_384

    if user_chunk is not None:
        chunk = user_chunk
    elif vram_gb >= 22:
        chunk = 196_608
    elif vram_gb >= 14:
        chunk = 131_072
    else:
        chunk = 65_536

    # 特征越宽，同样的 batch 需要的缓冲和激活值越多：按 (d_in + 1024) 反比压上限。
    # 常数 5.5e7 由实测确定：d_in=6144 时给约 7.7k token（8 GB 卡上稳定），
    # d_in=1024 时给约 27k token（仍低于 32k 的保守值）。
    if user_batch is None:
        width_cap = int(5.5e7 / (d_in + 1024))
        batch = max(4096, min(batch, width_cap))
    del d_out  # 目前只用于说明缓冲同时含 y，不参与上限计算

    return batch, chunk


def accumulate_train(
    reader: StreamReader,
    train_prompts: np.ndarray,
    chunk_tokens: int,
    device: str,
) -> RidgeAccumulator:
    """流式累加训练集的 XᵀX / XᵀY（读盘与 GPU 计算重叠）。

    Args:
        reader: 已打开的特征读取器。
        train_prompts: 训练集 prompt 索引（已排序；读取层也会再按磁盘偏移排序）。
        chunk_tokens: 每块的目标 token 数，只影响吞吐不影响结果。
        device: torch 设备。

    Returns:
        累加完成的 `RidgeAccumulator`。

    Notes:
        单纯「读一块算一块」会让 GPU 在每次读盘时空转：d_in=12288 时每块要读约 1.07 GB，
        读盘与 GEMM 大致各占一半，实测 GPU 利用率只在 10%~100% 之间来回跳喵。
        这里用两个主机缓冲槽 + 一个后台读盘线程，让第 N 块的 GEMM 与第 N+1 块的读盘重叠喵。
        槽的复用由 Event 握手：主线程算完才置位，读盘线程才敢覆盖喵。
        进程内存因此多占一倍（两个槽），仍与语料规模无关喵。
    """
    accumulator = RidgeAccumulator(reader.d_in, reader.d_out, drop_first=1, device=device)
    max_buf = chunk_tokens + 4096

    # 先把「每块包含哪些 prompt」规划好：读盘线程要能独立往下跑，不能边算边规划喵
    chunk_plan: list[list[int]] = []
    planned_tokens = 0
    p_idx = 0
    while p_idx < len(train_prompts):
        block: list[int] = []
        cur_tokens = 0
        while p_idx < len(train_prompts) and (cur_tokens == 0 or cur_tokens < chunk_tokens):
            pid = int(train_prompts[p_idx])
            block.append(pid)
            cur_tokens += int(reader.lengths[pid])
            p_idx += 1
        chunk_plan.append(block)
        planned_tokens += cur_tokens

    def make_slot() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.empty((max_buf, reader.d_in), dtype=np.float16),
            np.empty((max_buf, reader.d_out), dtype=np.float16),
        )

    slots = [make_slot(), make_slot()]
    free = [threading.Event(), threading.Event()]
    for event in free:
        event.set()
    chunk_q: "queue.Queue" = queue.Queue(maxsize=2)
    stop_ev = threading.Event()

    def producer() -> None:
        """后台读盘线程：按计划把 chunk 读进空闲槽并投递到队列。"""
        try:
            for index, block in enumerate(chunk_plan):
                if stop_ev.is_set():
                    break
                slot = index % 2
                free[slot].wait()
                free[slot].clear()
                if stop_ev.is_set():
                    break
                n = reader.read_chunk_into(block, slots[slot][0], slots[slot][1], drop_first=1)
                chunk_q.put((index, slot, n))
            chunk_q.put((None, None, None))  # 结束哨兵
        except BaseException as error:  # noqa: BLE001 - 线程异常必须抛回主线程
            chunk_q.put((None, None, error))

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()

    started = time.time()
    chunk_count = 0
    try:
        while True:
            index, slot, n = chunk_q.get()
            if index is None:
                if isinstance(n, BaseException):
                    raise n
                break
            try:
                if n > 0:
                    accumulator.add(slots[slot][0][:n], slots[slot][1][:n], clean=True)
                    chunk_count += 1
                    if chunk_count % 5 == 0:
                        trim_memory()
                        rss_gb, sys_avail_gb, vram_gb = get_memory_info()
                        check_ram_limit()
                        # 带上耗时与速率：只打 token 数无法判断快慢，容易把「正常」误读成「变慢」
                        elapsed = time.time() - started
                        rate = accumulator.n / elapsed if elapsed > 0 else 0.0
                        eta_min = (planned_tokens - accumulator.n) / rate / 60 if rate > 0 else 0.0
                        print(
                            f"  [累加] {accumulator.n:,} token | {elapsed:5.1f}s | {rate:7.0f} token/s"
                            f" | 剩余约 {eta_min:4.1f} 分"
                            f" | 进程 {rss_gb:.2f} GB | 可用: {sys_avail_gb:.2f} GB"
                            f" | VRAM: {vram_gb:.2f} GB",
                            flush=True,
                        )
            finally:
                free[slot].set()
    finally:
        stop_ev.set()
        for event in free:
            event.set()
        worker.join(timeout=5)

    print(
        f"累加完成: {accumulator.n:,} token，耗时 {time.time() - started:.1f}s",
        flush=True,
    )
    return accumulator


class TapSliceReader:
    """只暴露多 tap 特征里的某一个块。

    Args:
        reader: 原 `StreamReader`。
        block: 块下标（0 起），与 meta.json 里 `taps` 的顺序一致。
        n_blocks: 一共几个块。

    Notes:
        读取仍是整行读（磁盘 I/O 不变），只是把其余块丢掉喵。
        所以 d_in 变成块宽，累加器与显存随之按比例缩小喵。
        `d_in` 必须能被 `n_blocks` 整除，否则说明特征不是多 tap 拼接的。
    """

    def __init__(self, reader, block: int, n_blocks: int) -> None:
        if reader.d_in % n_blocks:
            raise ValueError(
                f"d_in={reader.d_in} 不能被块数 {n_blocks} 整除，无法按 tap 切分"
            )
        if not 0 <= block < n_blocks:
            raise ValueError(f"tap 块下标必须在 [0, {n_blocks})，得到 {block}")
        self._reader = reader
        self._block = block
        self._width = reader.d_in // n_blocks
        self.d_in = self._width
        self._buf_x: np.ndarray | None = None
        self._buf_y: np.ndarray | None = None

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def read_chunk_into(self, indices, x_buf, y_buf, drop_first: int = 0) -> int:
        need = int(sum(int(self._reader.lengths[i]) for i in indices))
        if self._buf_x is None or self._buf_x.shape[0] < need:
            self._buf_x = np.empty((need, self._reader.d_in), dtype=np.float16)
            self._buf_y = np.empty((need, self._reader.d_out), dtype=np.float16)
        n = self._reader.read_chunk_into(
            indices, self._buf_x, self._buf_y, drop_first=drop_first
        )
        start = self._block * self._width
        x_buf[:n] = self._buf_x[:n, start : start + self._width]
        y_buf[:n] = self._buf_y[:n]
        return n


def plan_ridge_chunk(
    device: str,
    d_in: int,
    d_out: int,
    user_chunk: int | None = None,
    mem_budget_gb: float | None = None,
) -> int:
    """按显存规划 Ridge 累加阶段的流式块大小。

    Args:
        device: `cuda` 或 `cpu`。
        d_in: 输入特征宽度（多 tap 拼接后是 1024 × 层数）。
        d_out: 目标宽度。
        user_chunk: 手动指定块大小；给了就直接用，不再自动推断。
        mem_budget_gb: 允许单块暂存占用的显存上限；None 时取显存的 25%。

    Returns:
        每块 token 数，范围 `[4096, 65536]`。

    Notes:
        单块峰值 = `chunk × (d_in + d_out) × 4` 字节——fp16 读入后要升 fp32 才能做 GEMM，
        它随 d_in **线性**增长：chunk=65536、d_in=6144 时是 2.5 GiB，
        d_in 翻到 12288 就是 4.0 GiB；再加上累加器 `d_in²×8`（1.6 GiB）就顶穿 8 GB 卡。
        这里按预算反推块大小，同时保留 `--chunk-tokens` 手动覆盖。
    """
    if user_chunk is not None:
        return int(user_chunk)
    if device == "cuda" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        vram_gb = 0.0
    budget_gb = mem_budget_gb if mem_budget_gb is not None else max(0.5, vram_gb * 0.25)
    per_token = max(1, (d_in + d_out) * 4)
    chunk = int(budget_gb * (1024**3) / per_token)
    return max(4096, min(65_536, chunk))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("dataset/features/v1"))
    parser.add_argument("--out", type=Path, default=Path("output/projection.safetensors"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("output/checkpoints"), help="检查点目录")
    parser.add_argument("--save-every-steps", type=int, default=25, help="每间隔 X 步保存一次检查点")
    parser.add_argument("--keep-last", type=int, default=3, help="只保留最近 X 个检查点（防止磁盘撑爆）")
    parser.add_argument("--resume", type=str, default=None, help="断点续训路径（或 'auto' 自动寻找最新检查点）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument(
        "--tap-index",
        type=int,
        default=None,
        help="只用多 tap 特征里的第 K 块（0 起，顺序同 meta.json 的 taps）",
    )
    parser.add_argument(
        "--mlp-buf-budget-gb",
        type=float,
        default=None,
        help="残差训练单侧 fp16 镜像的预算 GiB；默认按启动时可用内存自动定",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help="进程提交量熔断线 GiB；默认取物理内存的 65%%",
    )
    parser.add_argument("--batch", type=int, default=None, help="训练 batch 大小（token 数）。默认根据显存自动吃满")
    parser.add_argument("--chunk-tokens", type=int, default=None, help="流式读取缓冲区大小（token 数）")
    parser.add_argument("--eval-tokens", type=int, default=50_000, help="验证集评估 token 数量")
    parser.add_argument(
        "--mem-budget-gb",
        type=float,
        default=None,
        help="Ridge 单块暂存允许占用的显存 GiB；默认取显存的 25%%。手动覆盖用 --chunk-tokens",
    )
    parser.add_argument(
        "--recompute-ridge",
        action="store_true",
        help="忽略特征目录里的 Ridge 缓存，强制重新累加与 λ 扫描",
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="只解 Ridge 线性投影并导出，不训练残差 MLP",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="残差网络学习率（默认 1e-3；残差应远小于主体，过大会冲垮闭式解）"
    )
    parser.add_argument(
        "--eval-every-steps", type=int, default=0, help="轮内每 N 步评估一次验证余弦（0=关闭）"
    )
    parser.add_argument(
        "--linear-mode",
        choices=["joint", "frozen"],
        default=None,
        help="joint: W 与残差一起训（会拉偏闭式解）；frozen: W 冻结，只训残差（参考项目主配置）",
    )
    parser.add_argument("--smoke", action="store_true", help="小规模验证通道，不看指标")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = StreamReader(args.features)
    tap_blocks = json.loads((args.features / "meta.json").read_text(encoding="utf-8")).get("taps") or [-1]
    if args.tap_index is not None:
        tap_name = tap_blocks[args.tap_index] if args.tap_index < len(tap_blocks) else "?"
        print(f"只取 tap 块 #{args.tap_index}（对应层 {tap_name}）", flush=True)
        reader = TapSliceReader(reader, args.tap_index, len(tap_blocks))
    batch_size, chunk_tokens = auto_detect_hardware(
        device, args.batch, args.chunk_tokens, reader.d_in, reader.d_out
    )
    if args.smoke:
        batch_size = min(batch_size, 16_384)
        chunk_tokens = min(chunk_tokens, 65_536)

    rss_gb, sys_avail_gb, vram_gb = get_memory_info()
    print(
        f"特征: {reader.n_prompts} 条 / {reader.n_tokens:,} token / "
        f"{reader.d_in} -> {reader.d_out}  (设备: {device})",
        flush=True,
    )
    print(
        f"显存/批次自适应配置: Batch={batch_size:,} token | Chunk={chunk_tokens:,} token",
        flush=True,
    )

    # Ridge 累加阶段的块大小单独规划：它的暂存是 fp32 的，体积随 d_in 线性增长，
    # 不能沿用训练用的 chunk（那个只按显存档位挑，d_in 翻倍时会直接顶穿）。
    ridge_chunk = plan_ridge_chunk(
        device, reader.d_in, reader.d_out, args.chunk_tokens, args.mem_budget_gb
    )
    staging_gb = ridge_chunk * (reader.d_in + reader.d_out) * 4 / 1024**3
    accumulator_gb = (reader.d_in**2 * 8 + reader.d_in * reader.d_out * 8) / 1024**3
    solve_gb = (reader.d_in**2 * 4 * 2 + reader.d_in * reader.d_out * 4 * 3) / 1024**3
    peak_gb = max(staging_gb + accumulator_gb + reader.d_in**2 * 4 / 1024**3, accumulator_gb + solve_gb)
    print(
        f"Ridge 规划: Chunk={ridge_chunk:,} token | 暂存 {staging_gb:.2f} + 累加器 {accumulator_gb:.2f} GiB"
        f" | 预计峰值 {peak_gb:.2f} GiB",
        flush=True,
    )
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if peak_gb > vram_gb * 0.85:
            print(
                f"  [警告] 预计峰值 {peak_gb:.2f} GiB 超过显存 {vram_gb:.2f} GiB 的 85%%，"
                f"用 --chunk-tokens 或 --mem-budget-gb 调小",
                flush=True,
            )
    print(
        f"初始系统状态: 进程物理RAM: {rss_gb:.2f} GB | 系统可用RAM: {sys_avail_gb:.2f} GB | VRAM: {vram_gb:.2f} GB",
        flush=True,
    )

    splits = reader.split_by_prompt({"train": 0.70, "val": 0.15, "test": 0.15})
    print(
        f"划分: train {splits['train'].size} 条 / val {splits['val'].size} 条 / test {splits['test'].size} 条",
        flush=True,
    )

    train_prompts = np.sort(splits["train"])

    # 1/2. Ridge 累加与 λ 扫描：结果只取决于「特征库 + 口径」，跟之后的 MLP 实验无关，
    #      所以缓存在特征目录里，命中就整段跳过（4.5M token 约 100 秒 + 每个 λ 十几秒）。
    meta_bytes = (args.features / "meta.json").read_bytes()
    if args.tap_index is not None:
        meta_bytes += f"|tapblock={args.tap_index}".encode()
    stats_key = ridge_stats_key(meta_bytes, drop_first=1)
    stats_path = args.features / f"ridge_stats_{stats_key}.npz"

    lambdas = (1e-1, 1e1, 1e4) if args.smoke else LAMBDAS
    max_buf = ridge_chunk + 4096
    eval_target = min(max_buf, args.eval_tokens)
    solution_key = ridge_solution_key(stats_key, lambdas, eval_target)
    solution_path = args.features / f"ridge_solution_{solution_key}.npz"

    cached = None if args.recompute_ridge else load_solution(solution_path)
    if cached is None:
        accumulator = load_accumulator(stats_path, drop_first=1, device=device)
        if accumulator is None:
            print("\n--- 阶段 1: Ridge 矩阵累加 ---", flush=True)
            accumulator = accumulate_train(reader, train_prompts, ridge_chunk, device)
            save_accumulator(stats_path, accumulator)
            print(f"  累加器已缓存: {stats_path.name}", flush=True)
        else:
            print(f"\n--- 阶段 1: 命中累加缓存，跳过累加 ({accumulator.n:,} token) ---", flush=True)
            print(f"  缓存文件: {stats_path.name}", flush=True)

        # 验证集 λ 扫描：只新建一对缓冲，扫完立即释放
        print("\n--- 阶段 2: 正则系数 λ 扫描 ---", flush=True)
        val_prompts = np.sort(splits["val"])

        val_p_list: list[int] = []
        val_tokens_total = 0
        for pid in val_prompts:
            val_p_list.append(pid)
            val_tokens_total += int(reader.lengths[pid])
            if val_tokens_total >= eval_target:
                break

        x_buf = np.empty((max_buf, reader.d_in), dtype=np.float16)
        y_buf = np.empty((max_buf, reader.d_out), dtype=np.float16)
        v_n = reader.read_chunk_into(val_p_list, x_buf, y_buf, drop_first=1)
        val_x = x_buf[:v_n]
        val_y = y_buf[:v_n]

        scored: list[tuple[float, float, float, object]] = []
        for lam in lambdas:
            t0 = time.time()
            solution = accumulator.solve(lam)
            cos, rel = solution.score(val_x, val_y, chunk=ridge_chunk, device=device)
            scored.append((lam, cos, rel, solution))
            print(f"  λ {lam:8.0e} | cos {cos:.5f} | relL2 {rel:.4f} ({time.time() - t0:.2f}s)", flush=True)

        train_token_count = int(accumulator.n)
        del x_buf, y_buf, val_x, val_y, accumulator
        trim_memory()

        grid = [(lam, cos, rel) for lam, cos, rel, _ in scored]
        best_index = int(np.argmax([row[1] for row in scored]))
        best_lam, best_cos, best_rel, best_solution = scored[best_index]
        save_solution(solution_path, best_solution, grid)
        print(f"  解已缓存: {solution_path.name}", flush=True)
    else:
        best_solution, grid = cached
        best_lam = best_solution.lam
        best_cos, best_rel = next(
            (cos, rel) for lam, cos, rel in grid if lam == best_lam
        )
        train_token_count = int(best_solution.n_tokens)
        print("\n--- 阶段 1/2: 命中 Ridge 缓存，跳过累加与 λ 扫描 ---", flush=True)
        print(f"  缓存文件: {solution_path.name}", flush=True)
        for lam, cos, rel in grid:
            print(f"  λ {lam:8.0e} | cos {cos:.5f} | relL2 {rel:.4f} (缓存)", flush=True)

    print(f"\n最佳 λ: {best_lam:.0e} | 验证集余弦: {best_cos:.5f} | relL2: {best_rel:.4f}", flush=True)

    # 立即导出纯线性 Ridge 适配器，保证硬盘上始终有立即可用的基础模型
    linear_out = args.out.parent / "projection_linear.safetensors"
    # 抽头口径必须写进 metadata：推理节点靠它知道取哪几层、按什么顺序拼接
    features_meta = json.loads((args.features / "meta.json").read_text(encoding="utf-8"))
    raw_taps = features_meta.get("taps") or [features_meta.get("tap", -1)]
    linear_meta = {
        "tap": features_meta.get("tap", -1),
        "taps": ",".join(str(int(tap)) for tap in raw_taps),
        "d_in": int(reader.d_in),
        "train_prompts": int(splits["train"].size),
        "train_tokens": train_token_count,
        "val_cosine": f"{best_cos:.6f}",
        "rel_l2": f"{best_rel:.6f}",
        "source": "qwen3.5-0.8b",
        "target": "qwen3.5-9b",
    }
    save_linear(linear_out, best_solution, meta=linear_meta)
    print(f"【纯线性 Ridge 基础权重已就绪】: {linear_out}", flush=True)

    if args.linear_only:
        # 0.8B 上残差网络实测只涨 +0.00014（1e-4 量级），训练它没有意义。
        # 这个开关让「产出最终适配器」变成一次纯闭式解，几秒到两分钟。
        print("\n--linear-only：跳过残差 MLP 训练，产物已导出", flush=True)
        return 0

    # 3. 残差 MLP 训练：利用自适应 batch 吃满 GPU 算力加速训练
    print("\n--- 阶段 3: 残差网络训练 (MLP) ---", flush=True)
    config = MLPConfig()
    config.batch = batch_size
    config.chunk_tokens = chunk_tokens
    config.buf_budget_gb = args.mlp_buf_budget_gb
    config.max_process_ram_gb = args.max_ram_gb
    config.checkpoint_dir = str(args.checkpoint_dir)
    config.save_every_steps = args.save_every_steps
    config.keep_last_checkpoints = args.keep_last
    config.resume = args.resume
    if args.smoke:
        config.hidden = 2048
        config.depth = 2
        config.epochs = 3
        config.patience = 3
        config.eval_tokens = 50_000
    else:
        config.eval_tokens = args.eval_tokens

    if args.hidden is not None:
        config.hidden = args.hidden
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.linear_mode is not None:
        config.linear_mode = args.linear_mode
    if args.lr is not None:
        config.lr = args.lr
    if args.eval_every_steps:
        config.eval_every_steps = args.eval_every_steps

    print(
        f"网络配置: hidden={config.hidden}, depth={config.depth}, batch={config.batch}, epochs={config.epochs}",
        flush=True,
    )

    result = train(reader, best_solution, config, log=lambda text: print(text, flush=True))
    print(
        f"\n训练结束 ({result.epochs_run} 轮): Ridge 基准 {result.base_val_cosine:.5f} -> "
        f"MLP {result.best_val_cosine:.5f} (净提升: {result.improvement():+.5f})",
        flush=True,
    )

    # 4. 导出 safetensors 适配器
    meta = {
        "tap": json.loads((args.features / "meta.json").read_text(encoding="utf-8")).get("tap", -1),
        "train_prompts": int(splits["train"].size),
        "train_tokens": train_token_count,
        "val_cosine": f"{result.best_val_cosine:.6f}",
        "source": "qwen3.5-0.8b",
        "target": "qwen3.5-9b",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, best_solution, result, config, meta=meta)
    print(f"产物已导出: {args.out}", flush=True)

    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
