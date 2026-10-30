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
    batch: int = 8192
    lr: float = 1e-3
    weight_decay: float = 0.01
    patience: int = 8
    seed: int = 20261027
    eval_tokens: int = 200_000
    amp: bool = True
    compile: bool = True
    compile_mode: str = "default"
    skip: str = "ridge"
    # "joint"：W 与 MLP 一起训，W 从 Ridge 解出发（第 0 步等价于 Ridge）。
    # "frozen"：W 固定为 Ridge 解，只训残差（参考项目主配置）。
    linear_mode: str = "joint"
    linear_lr_scale: float = 0.1


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


def _cosine_rows(pred, target) -> float:
    """分批计算逐行余弦均值，避免一次算太大。

    Args:
        pred: `[n, d]` 预测。
        target: `[n, d]` 目标。

    Returns:
        均值余弦。
    """
    torch, _ = _require_torch()
    dot = (pred * target).sum(1)
    norm = pred.norm(dim=1) * target.norm(dim=1)
    valid = norm > 0
    if not bool(valid.any()):
        return 0.0
    return float((dot[valid] / norm[valid]).mean())


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    def to_tensors(indices: np.ndarray):
        """把 token 下标取成标准化后的输入与标准化空间的目标。

        Args:
            indices: 一维 token 下标数组。

        Returns:
            `(z, y_std, y_raw)`，前两个用于训练，`y_raw` 用于验证时回原空间。
        """
        x = torch.from_numpy(np.asarray(store.student[indices], dtype=np.float32))
        y = torch.from_numpy(np.asarray(store.teacher[indices], dtype=np.float32))
        x = x.to(device)
        y = y.to(device)
        return (x - mean_in) / std_in, (y - mean_out) / std_out, y

    def evaluate() -> float:
        """在验证 token 上算原始空间余弦。

        Returns:
            平均余弦。
        """
        net.eval()
        cos_sum = 0.0
        total_n = 0
        with torch.no_grad():
            for start in range(0, val_idx.size, 16384):
                chunk = val_idx[start : start + 16384]
                z, _, y_raw = to_tensors(chunk)
                with autocast():
                    pred_std = (z @ W_used if use_ridge else 0) + net(z)
                # 余弦在 fp32 里算：bf16 只有 8 位尾数，算距离会明显失真。
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
    history: list[tuple[int, float, float]] = []
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
                # 损失在 fp32 里算：bf16 的尾数不够，MSE 会被量化误差带偏。
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
