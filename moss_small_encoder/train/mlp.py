"""非线性投影：在 Ridge 之上叠一个零初始化残差网络。

0.8B 只有 1024 维，纯线性映射能表达的东西有限（参考项目实测线性在 4B 上就饱和，
数据翻 8 倍只涨 1.8%），所以这一步的收益主要来自非线性。

结构（全部在 Ridge 的标准化空间里做）：

    z      = (h - mean_in) / std_in
    pred   = z @ W_ridge + MLP(z)
    MLP    = Linear(d_in, h) -> GELU -> [Linear(h, h) -> GELU] * (depth-1) -> Linear(h, d_out)

**末层 weight 与 bias 全部零初始化**，所以第 0 步的输出严格等于 Ridge 的结果。
训练失败最多退回线性解，不会比它更差。

早停按 **prompt** 分组评估，不按 token：同一 prompt 的 token 高度相关，
按 token 切验证集会把训练信号泄漏进验证，指标虚高。
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ridge import check_ram_limit, get_memory_info, trim_memory

torch = None
nn = None


def _require_torch():
    """延迟导入 torch，让不需要训练的环境（只建数据集）不必装它。

    Returns:
        `(torch, torch.nn)` 模块元组。

    Raises:
        ImportError: 没装 torch 时给出可执行的提示。
    """
    global torch, nn
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn
        except ImportError as error:  # pragma: no cover - 环境问题
            raise ImportError(
                "训练需要 torch：pip install -r requirements-encode.txt"
            ) from error
        torch, nn = _torch, _nn
    return torch, nn


@dataclass
class MLPConfig:
    """残差网络与训练超参数。

    Attributes:
        hidden: 隐藏层宽度。
        depth: 隐藏层个数（1 表示单隐层）。
        epochs: 最大轮数。
        batch: 每步 token 数。
        lr: 学习率。
        weight_decay: AdamW 的权重衰减。
        patience: 验证指标连续多少轮不涨就停。
        seed: 随机种子。
        eval_tokens: 验证时最多用多少 token（控制时间）。
    """

    hidden: int = 8192
    depth: int = 2
    epochs: int = 60
    batch: int = 24576
    lr: float = 1e-3
    weight_decay: float = 0.01
    patience: int = 8
    seed: int = 20261027
    eval_tokens: int = 100_000
    amp: bool = True
    compile: bool = True
    compile_mode: str = "default"
    skip: str = "ridge"
    # "joint"：W 与 MLP 一起训，W 从 Ridge 解出发（第 0 步等价于 Ridge）。
    # "frozen"：W 固定为 Ridge 解，只训残差（参考项目主配置）。
    linear_mode: str = "joint"
    linear_lr_scale: float = 0.1
    checkpoint_dir: str | None = "output/checkpoints"
    save_every_steps: int = 25
    keep_last_checkpoints: int = 3
    resume: str | None = None
    min_lr_ratio: float = 0.01
    min_sys_available_gb: float = 2.0
    max_process_ram_gb: float = 14.0


def build(d_in: int, d_out: int, hidden: int, depth: int, zero_init: bool = True):
    """构造残差网络。

    Args:
        d_in: 输入维度（student 隐状态维度）。
        d_out: 输出维度（teacher 隐状态维度）。
        hidden: 隐藏层宽度。
        depth: 隐藏层个数，>= 1。
        zero_init: 末层是否零初始化。

    Returns:
        `torch.nn.Sequential`，输入已标准化的 `[n, d_in]`，输出残差增量 `[n, d_out]`。

    Raises:
        ValueError: depth < 1。
    """
    torch, nn = _require_torch()
    if depth < 1:
        raise ValueError("depth 必须 >= 1")
    layers: list = [nn.Linear(d_in, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, d_out))
    net = nn.Sequential(*layers)
    if zero_init:
        last = net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return net


@dataclass
class TrainResult:
    """训练结果。

    Attributes:
        best_val_cosine: 验证集上最好的余弦。
        base_val_cosine: 同为验证集时纯 Ridge 的余弦（对照）。
        epochs_run: 实际跑了多少轮。
        state: 最好那轮的 state_dict（CPU 张量）。
        history: 每轮 `(epoch, train_loss, val_cosine)`。
    """

    best_val_cosine: float
    base_val_cosine: float
    epochs_run: int
    state: dict
    W: np.ndarray | None = None
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def improvement(self) -> float:
        """相对纯 Ridge 的余弦提升。

        Returns:
            绝对提升值，正数表示非线性确实有用。
        """
        return self.best_val_cosine - self.base_val_cosine


class CheckpointManager:
    """按步数间隔保存检查点，并只保留最近 N 个检查点（防止磁盘撑爆）。"""

    def __init__(self, directory: Path | str, save_every_steps: int = 25, keep_last: int = 3):
        self.directory = Path(directory)
        self.save_every_steps = int(save_every_steps)
        self.keep_last = int(keep_last)
        self.saved_ckpts: list[Path] = []
        self.directory.mkdir(parents=True, exist_ok=True)

    def should_save(self, global_step: int) -> bool:
        return self.save_every_steps > 0 and (global_step > 0 and global_step % self.save_every_steps == 0)

    def save(
        self,
        global_step: int,
        epoch: int,
        net,
        optimizer,
        scheduler,
        extra: dict | None = None,
    ) -> Path:
        raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
        ckpt_path = self.directory / f"checkpoint_step_{global_step:06d}.pt"
        data = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state": {k: v.cpu() for k, v in raw_net.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            **(extra or {}),
        }
        torch.save(data, ckpt_path)
        torch.save(data, self.directory / "checkpoint_latest.pt")
        self.saved_ckpts.append(ckpt_path)

        if self.keep_last > 0 and len(self.saved_ckpts) > self.keep_last:
            to_remove = self.saved_ckpts[:-self.keep_last]
            for p in to_remove:
                try:
                    if p.exists() and p != ckpt_path:
                        p.unlink()
                except OSError:
                    pass
            self.saved_ckpts = self.saved_ckpts[-self.keep_last:]

        return ckpt_path

    def save_best(
        self,
        global_step: int,
        epoch: int,
        net,
        val_cosine: float,
        best_cosine: float,
        extra: dict | None = None,
    ) -> Path:
        raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
        best_path = self.directory / "checkpoint_best.pt"
        data = {
            "global_step": global_step,
            "epoch": epoch,
            "model_state": {k: v.cpu() for k, v in raw_net.state_dict().items()},
            "val_cosine": val_cosine,
            "best_cosine": best_cosine,
            **(extra or {}),
        }
        torch.save(data, best_path)
        return best_path


def _cosine_rows(pred, target, block_size: int = 8192) -> float:
    """分批计算逐行余弦均值，避免一次分配过大中间张量。

    Args:
        pred: `[n, d]` 预测。
        target: `[n, d]` 目标。
        block_size: 分批计算步长，默认 8192 行（显存占用 <140MB）。

    Returns:
        均值余弦。
    """
    torch, _ = _require_torch()
    n = pred.shape[0]
    if n == 0:
        return 0.0
    if n <= block_size:
        dot = (pred * target).sum(1)
        norm = pred.norm(dim=1) * target.norm(dim=1)
        valid = norm > 0
        if not bool(valid.any()):
            return 0.0
        return float((dot[valid] / norm[valid]).mean())

    total_cos = 0.0
    valid_count = 0
    for i in range(0, n, block_size):
        end = min(i + block_size, n)
        p_sub = pred[i:end]
        t_sub = target[i:end]
        dot = (p_sub * t_sub).sum(1)
        norm = p_sub.norm(dim=1) * t_sub.norm(dim=1)
        valid = norm > 0
        if bool(valid.any()):
            total_cos += float((dot[valid] / norm[valid]).sum())
            valid_count += int(valid.sum())
    return total_cos / max(1, valid_count)


def train(
    store,
    solution,
    config: MLPConfig | None = None,
    device: str | None = None,
    log=print,
) -> TrainResult:
    """训练残差网络。

    Args:
        store: `moss_small_encoder.train.store.FeatureStore`。
        solution: `ridge.RidgeSolution`，提供 W 与标准化参数，也作为残差基线。
        config: 超参数；None 用默认值。
        device: `"cuda"` / `"cpu"`；None 自动选。
        log: 日志函数。

    Returns:
        TrainResult，`state` 是最好那轮的权重。

    Notes:
        - 目标在标准化空间里算 MSE，等价于按 teacher 各维方差加权，比直接在原始空间
          算 MSE 稳定：teacher 的 attention sink 维方差极大，不标准化会主导梯度。
        - 验证指标回到原始空间算余弦，与最终导出后的实际行为一致。
    """
    torch, nn = _require_torch()
    config = config or MLPConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(config.seed)
    # skip="ridge"：参考项目主配置，输出 = 线性 + 残差，末层零初始化。
    # skip="zero"：参考项目 mlp.py 里的消融档，去掉线性、末层不零初始化
    #   （零初始化会让第一层梯度为 0，网络根本起不来）。
    use_ridge = config.skip != "zero"
    # bf16 混合精度：前向/反向用 bf16 走张量核，主权重与优化器状态仍是 fp32。
    # Ada（4060）对 bf16 支持好，且比 fp16 不容易溢出（指数位与 fp32 相同）。
    use_amp = bool(config.amp) and device.startswith("cuda")

    def autocast():
        """返回一个上下文管理器：CUDA 上启用 bf16 autocast，CPU 上什么都不做。

        Returns:
            `contextlib.nullcontext` 或 `torch.autocast`。
        """
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    splits = store.split_by_prompt({"train": 0.70, "val": 0.15, "test": 0.15})
    train_idx = store.token_indices(splits["train"])
    val_idx = store.token_indices(splits["val"])
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train 或 val 为空，检查划分比例与样本数")

    if val_idx.size > config.eval_tokens:
        val_idx = np.sort(rng.choice(val_idx, size=config.eval_tokens, replace=False))
    # 验证也走固定块大小，否则 compile 会为每个尾块重新编译一次。
    eval_chunk = 16384
    val_idx = val_idx[: (val_idx.size // eval_chunk) * eval_chunk] if val_idx.size >= eval_chunk else val_idx

    # 两个 W：W_ref 是 Ridge 原解，永远用作基线对照；W_used 是实际参与前向的那个。
    W_ref = torch.from_numpy(solution.W).to(device)
    joint = config.linear_mode == "joint" and use_ridge
    W_used = torch.nn.Parameter(W_ref.clone()) if joint else W_ref
    mean_in = torch.from_numpy(solution.mean_in).to(device)
    std_in = torch.from_numpy(solution.std_in).to(device)
    mean_out = torch.from_numpy(solution.mean_out).to(device)
    std_out = torch.from_numpy(solution.std_out).to(device)

    net = build(
        store.d_in, store.d_out, config.hidden, config.depth, zero_init=use_ridge
    ).to(device)
    raw_net = net  # compile 后 state_dict 的键会带 `_orig_mod.` 前缀，取权重要用原对象
    if config.compile and hasattr(torch, "compile"):
        # Windows 上 torch.compile 依赖 Triton（triton-windows）。装不上就退回 eager：
        # 训练正确性不依赖它，只是 kernel 启动开销更大。
        try:
            net = torch.compile(net, mode=config.compile_mode)
            log("  torch.compile 已启用")
        except Exception as error:  # noqa: BLE001 - 后端差异太多，统一降级
            log(f"  torch.compile 不可用，退回 eager：{error}")
    params = net.parameters()
    if joint:
        # W 已经是最优解，给它更小的学习率，并且**不加 weight decay**：
        # AdamW 的解耦衰减会持续把 W 往零拉，那是纯粹的破坏。
        groups = [
            {"params": [W_used], "weight_decay": 0.0, "lr": config.lr * config.linear_lr_scale},
            {"params": list(params), "weight_decay": config.weight_decay, "lr": config.lr},
        ]
        optimizer = torch.optim.AdamW(groups)
        clip_params = [W_used] + list(params)
    else:
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        clip_params = list(params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * config.min_lr_ratio
    )

    ckpt_mgr = (
        CheckpointManager(
            config.checkpoint_dir,
            save_every_steps=config.save_every_steps,
            keep_last=config.keep_last_checkpoints,
        )
        if config.checkpoint_dir
        else None
    )
    start_epoch = 0
    global_step = 0

    if config.resume:
        resume_p = Path(config.resume)
        if resume_p.is_dir():
            resume_p = resume_p / "checkpoint_latest.pt"
        if resume_p.exists():
            log(f"  [Resume] 恢复检查点: {resume_p}")
            ckpt = torch.load(resume_p, map_location=device)
            raw_net.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            log(f"  [Resume] 成功从 Epoch {start_epoch} (全局步 {global_step}) 继续训练")

    is_stream = hasattr(store, "read_chunk_into")

    if is_stream:
        # ---- 真流式读取分支：复用单一缓冲区，内存锁死在安全线，GPU 显存吃满 ----
        chunk_token_target = 65_536
        buf_capacity = chunk_token_target + config.batch + 8192
        buf_x = np.empty((buf_capacity, store.d_in), dtype=np.float16)
        buf_y = np.empty((buf_capacity, store.d_out), dtype=np.float16)

        val_prompts = np.sort(splits["val"])
        val_target = min(config.eval_tokens, int(store.lengths[val_prompts].sum()) - len(val_prompts))
        val_p_list: list[int] = []
        cur_v = 0
        for p in val_prompts:
            val_p_list.append(p)
            cur_v += int(store.lengths[p]) - 1
            if cur_v >= val_target:
                break

        def evaluate(use_ref: bool = False) -> float:
            target_name = "Ridge 基准" if use_ref else "模型验证"
            log(f"  [Eval] 正在评估{target_name} (目标 {val_target:,} tokens)...")
            eval_t0 = time.time()
            net.eval()
            cos_sum = 0.0
            total_n = 0
            with torch.no_grad():
                v_idx = 0
                while v_idx < len(val_p_list):
                    v_batch: list[int] = []
                    v_cur = 0
                    while v_idx < len(val_p_list) and (v_cur == 0 or v_cur < chunk_token_target):
                        pid = int(val_p_list[v_idx])
                        v_batch.append(pid)
                        v_cur += int(store.lengths[pid])
                        v_idx += 1
                    n_v = store.read_chunk_into(v_batch, buf_x, buf_y, drop_first=1)
                    if n_v < 1:
                        continue
                    # 分 batch 走模型推理，防止一次性 6.5 万 token 撑爆显存
                    for b_start in range(0, n_v, config.batch):
                        b_end = min(b_start + config.batch, n_v)
                        b_size = b_end - b_start
                        if b_size < 1:
                            continue
                        xb = torch.from_numpy(buf_x[b_start:b_end]).to(device).float()
                        yb = torch.from_numpy(buf_y[b_start:b_end]).to(device).float()
                        z = (xb - mean_in) / std_in
                        with autocast():
                            if use_ref:
                                pred_std = z @ W_ref
                            else:
                                pred_std = (z @ W_used if use_ridge else 0) + net(z)
                        pred_raw = pred_std.float() * std_out + mean_out
                        cos_sum += _cosine_rows(pred_raw, yb) * b_size
                        total_n += b_size
                        del xb, yb, z, pred_std, pred_raw
            net.train()
            val_cos = cos_sum / max(1, total_n)
            log(f"  [Eval] {target_name}评估完成: 余弦 {val_cos:.5f} | 耗时 {time.time() - eval_t0:.1f}s")
            return val_cos

        base_cosine = evaluate(use_ref=True)
        best = base_cosine
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = -1
        stale = 0
        history: list[tuple[int, float, float]] = []
        started = time.time()

        train_p_all = np.sort(splits["train"])
        train_total_tokens = sum(int(store.lengths[p]) - 1 for p in train_p_all)
        total_steps = max(1, (train_total_tokens + config.batch - 1) // config.batch)
        train_x_buf = buf_x
        train_y_buf = buf_y

        for epoch in range(start_epoch, config.epochs):
            epoch_t0 = time.time()
            # 严格单向纯流式顺序读取：按磁盘物理位置升序遍历，绝对不打乱 prompt 顺序！
            p_order = train_p_all
            running = 0.0
            steps = 0
            p_idx = 0
            leftover_n = 0

            step_t0 = time.time()
            recent_tokens = 0
            recent_steps = 0

            while p_idx < len(p_order) or leftover_n > 0:
                # 1. 只要还有 prompt，就拉入下一个 chunk 追加在未成 batch 的尾数后面
                if p_idx < len(p_order):
                    p_batch: list[int] = []
                    cur_tokens = 0
                    while p_idx < len(p_order) and (cur_tokens == 0 or cur_tokens < chunk_token_target):
                        pid = int(p_order[p_idx])
                        p_batch.append(pid)
                        cur_tokens += int(store.lengths[pid])
                        p_idx += 1

                    n_read = store.read_chunk_into(
                        p_batch, train_x_buf, train_y_buf, drop_first=1, offset=leftover_n
                    )
                    total_avail = leftover_n + n_read
                else:
                    total_avail = leftover_n
                    n_read = 0

                if total_avail < 2:
                    break

                # 2. 如果已到本轮所有 prompt 的终点，允许最后一步小于完整 batch
                is_epoch_end = (p_idx >= len(p_order))
                if is_epoch_end:
                    num_batches = (total_avail + config.batch - 1) // config.batch
                else:
                    num_batches = total_avail // config.batch

                # 3. 逐 batch 执行梯度更新，中间所有批次大小均严格等于 config.batch
                for b in range(num_batches):
                    b_start = b * config.batch
                    b_end = min(b_start + config.batch, total_avail)
                    b_size = b_end - b_start
                    if b_size < 2:
                        continue

                    xt = torch.from_numpy(train_x_buf[b_start:b_end]).to(device).float()
                    yt = torch.from_numpy(train_y_buf[b_start:b_end]).to(device).float()
                    perm = torch.randperm(b_size, device=device)
                    z = ((xt - mean_in) / std_in)[perm]
                    y_std = ((yt - mean_out) / std_out)[perm]
                    del xt, yt

                    if epoch == 0 and steps == 0 and config.compile:
                        log("  [JIT] 正在首次编译计算图 (torch.compile，首次耗时约几十秒)...")

                    with autocast():
                        pred_std = (z @ W_used if use_ridge else 0) + net(z)
                        loss = torch.nn.functional.mse_loss(pred_std.float(), y_std)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                    optimizer.step()

                    loss_val = float(loss.detach())
                    running += loss_val
                    steps += 1
                    global_step += 1
                    recent_tokens += b_size
                    recent_steps += 1
                    del z, y_std, pred_std, loss

                    if ckpt_mgr and ckpt_mgr.should_save(global_step):
                        ckpt_p = ckpt_mgr.save(
                            global_step, epoch, raw_net, optimizer, scheduler, {"loss": loss_val}
                        )
                        log(
                            f"  [Checkpoint] 旋转检查点已存盘 (步 {global_step}, 保留最近 {ckpt_mgr.keep_last} 个): {ckpt_p.name}"
                        )

                    if steps == 1 or steps % 25 == 0 or steps == total_steps:
                        dt = time.time() - step_t0
                        cur_lr = optimizer.param_groups[-1]["lr"]
                        rss_gb, sys_avail_gb, vram_gb = get_memory_info()
                        tok_s = recent_tokens / max(1e-4, dt)
                        ms_step = (dt / max(1, recent_steps)) * 1000
                        log(
                            f"  [E{epoch:02d}|步 {steps:3d}/{total_steps}] loss: {loss_val:.5f} | "
                            f"lr: {cur_lr:.2e} | {ms_step:.1f}ms/步 ({tok_s/1e3:.0f}k tok/s) | "
                            f"RAM: {rss_gb:.2f}GB (可用 {sys_avail_gb:.1f}GB) | VRAM: {vram_gb:.2f}GB"
                        )
                        step_t0 = time.time()
                        recent_tokens = 0
                        recent_steps = 0

                # 4. 若未到轮末，将未满整 batch 的余数结转到 buffer 起始处
                if not is_epoch_end:
                    processed = num_batches * config.batch
                    leftover_n = total_avail - processed
                    if leftover_n > 0:
                        train_x_buf[:leftover_n] = train_x_buf[processed:total_avail]
                        train_y_buf[:leftover_n] = train_y_buf[processed:total_avail]
                else:
                    leftover_n = 0

            scheduler.step()
            val_cosine = evaluate()
            mean_loss = running / max(1, steps)
            history.append((epoch, mean_loss, val_cosine))

            trim_memory()
            check_ram_limit(config.min_sys_available_gb, config.max_process_ram_gb)

            if val_cosine > best + 1e-6:
                best = val_cosine
                best_state = copy.deepcopy(raw_net.state_dict())
                best_epoch = epoch
                stale = 0
                if ckpt_mgr:
                    ckpt_mgr.save_best(
                        global_step, epoch, raw_net, val_cosine, best, {"mean_loss": mean_loss}
                    )
                    log(f"  [Checkpoint] 最佳检查点已更新: checkpoint_best.pt (余弦: {best:.5f})")
            else:
                stale += 1

            if ckpt_mgr:
                ckpt_mgr.save(
                    global_step, epoch, raw_net, optimizer, scheduler,
                    {"val_cosine": val_cosine, "mean_loss": mean_loss}
                )

            log(
                f"  ==> Epoch {epoch:02d} 完成 | 平均 loss: {mean_loss:.6f} | 验证集余弦: {val_cosine:.5f}"
                f" (最佳: {best:.5f}) | 耗时: {time.time() - epoch_t0:.1f}s"
            )
            if stale >= config.patience:
                log(f"  早停：{config.patience} 轮无提升（最好在第 {best_epoch} 轮）")
                break

        del train_x_buf, train_y_buf, buf_x, buf_y
        trim_memory()

    else:
        # ---- 兼容普通 FeatureStore（单元测试小数据集用）----
        def to_tensors(indices: np.ndarray):
            x = (
                torch.from_numpy(np.ascontiguousarray(store.student[indices]))
                .to(device)
                .float()
            )
            y = (
                torch.from_numpy(np.ascontiguousarray(store.teacher[indices]))
                .to(device)
                .float()
            )
            return (x - mean_in) / std_in, (y - mean_out) / std_out, y

        def evaluate() -> float:
            net.eval()
            cos_sum = 0.0
            total_n = 0
            with torch.no_grad():
                for start in range(0, val_idx.size, 16384):
                    chunk = val_idx[start : start + 16384]
                    z, _, y_raw = to_tensors(chunk)
                    with autocast():
                        pred_std = (z @ W_used if use_ridge else 0) + net(z)
                    pred_raw = pred_std.float() * std_out + mean_out
                    cos_sum += _cosine_rows(pred_raw, y_raw) * chunk.size
                    total_n += int(chunk.size)
            net.train()
            return cos_sum / max(1, total_n)

        base_cosine = 0.0
        with torch.no_grad():
            net.eval()
            cos_sum = 0.0
            total_n = 0
            for start in range(0, val_idx.size, 16384):
                chunk = val_idx[start : start + 16384]
                z, _, y_raw = to_tensors(chunk)
                with autocast():
                    pred_std = z @ W_ref
                pred_raw = pred_std.float() * std_out + mean_out
                cos_sum += _cosine_rows(pred_raw, y_raw) * chunk.size
                total_n += int(chunk.size)
            base_cosine = cos_sum / max(1, total_n)
        net.train()

        best = base_cosine
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = -1
        stale = 0
        history = []
        started = time.time()

        for epoch in range(config.epochs):
            order = rng.permutation(train_idx)
            running = 0.0
            steps = 0
            for start in range(0, order.size, config.batch):
                chunk = order[start : start + config.batch]
                if chunk.size < 2:
                    continue
                z, y_std, _ = to_tensors(chunk)
                with autocast():
                    pred_std = (z @ W_used if use_ridge else 0) + net(z)
                    loss = torch.nn.functional.mse_loss(pred_std.float(), y_std)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()
                running += float(loss.detach())
                steps += 1
            scheduler.step()
            val_cosine = evaluate()
            mean_loss = running / max(1, steps)
            history.append((epoch, mean_loss, val_cosine))
            trim_memory()
            check_ram_limit(config.min_sys_available_gb, config.max_process_ram_gb)
            if val_cosine > best + 1e-6:
                best = val_cosine
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if epoch % 5 == 0 or stale == 0:
                log(
                    f"  epoch {epoch:3d}  loss {mean_loss:.6f}  val_cos {val_cosine:.5f}"
                    f"  best {best:.5f}  ({time.time() - started:.0f}s)"
                )
            if stale >= config.patience:
                log(f"  早停：{config.patience} 轮无提升（最好在第 {best_epoch} 轮）")
                break

    trained_W = (
        W_used.detach().float().cpu().numpy() if joint else solution.W
    )
    return TrainResult(
        W=trained_W,
        best_val_cosine=best,
        base_val_cosine=base_cosine,
        epochs_run=len(history),
        state=best_state,
        history=history,
    )


def save_linear(path: Path, solution, meta: dict | None = None) -> None:
    """把纯线性 Ridge 参数导出成对标原参考项目的 safetensors。

    Args:
        path: 输出路径。
        solution: Ridge 解。
        meta: 额外元数据。
    """
    torch, _ = _require_torch()
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError("导出需要 safetensors：pip install safetensors") from error

    tensors = {
        "weight": torch.from_numpy(solution.W).half().contiguous(),
        "mean_in": torch.from_numpy(solution.mean_in).float().contiguous(),
        "std_in": torch.from_numpy(solution.std_in).float().contiguous(),
        "mean_out": torch.from_numpy(solution.mean_out).float().contiguous(),
        "std_out": torch.from_numpy(solution.std_out).float().contiguous(),
    }
    metadata = {
        "format": "potatoforge_moss_encoder_projection_v1",
        "source_layer": "last_normalized",
        "target_layer": "last_normalized",
        "alpha": str(solution.lam),
    }
    if meta:
        metadata.update({k: str(v) for k, v in meta.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    Path(path).with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save(
    path: Path,
    solution,
    result: TrainResult,
    config: MLPConfig,
    meta: dict | None = None,
) -> None:
    """把线性 + 非线性参数导出成一个 safetensors。

    Args:
        path: 输出路径。
        solution: Ridge 解。
        result: 训练结果（提供残差网络权重）。
        config: 网络结构超参数，导出时必须一起存，否则无法重建。
        meta: 额外元信息，例如 tap、语料规模、验证余弦。

    Returns:
        无。

    Raises:
        ImportError: 没装 safetensors 时提示。

    Notes:
        按参考项目的**实际发布文件**定的精度（不是我猜的）：
        `weight` 与 MLP 权重存 **float16**，归一化统计量（mean/std）存 **float32**。
        统计量只有一万多个数，用 fp32 几乎不占空间；权重是大头，用 fp16 省一半。
        参考项目对 W 做过等价性检查：fp32 只是把文件和显存翻倍，余弦差在 0.999993。
    """
    torch, _ = _require_torch()
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise ImportError("导出需要 safetensors：pip install safetensors") from error

    # 联合训练时用的是训练后的 W；只训残差时就是 Ridge 原解。
    W_out = result.W if result.W is not None else solution.W
    tensors = {
        "weight": torch.from_numpy(W_out).half().contiguous(),
        "mean_in": torch.from_numpy(solution.mean_in).float().contiguous(),
        "std_in": torch.from_numpy(solution.std_in).float().contiguous(),
        "mean_out": torch.from_numpy(solution.mean_out).float().contiguous(),
        "std_out": torch.from_numpy(solution.std_out).float().contiguous(),
    }
    for name, value in result.state.items():
        tensors[f"mlp.{name}"] = value.detach().half().contiguous()

    metadata = {
        "format": "moss_encoder_projection_v1",
        "source_layer": "last_normalized",
        "target_layer": "last_normalized",
        "hidden": str(config.hidden),
        "depth": str(config.depth),
        "alpha": str(solution.lam),
        "val_cosine": f"{result.best_val_cosine:.6f}",
        "val_cosine_ridge": f"{result.base_val_cosine:.6f}",
        "epochs": str(result.epochs_run),
    }
    if meta:
        metadata.update({k: str(v) for k, v in meta.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    Path(path).with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
