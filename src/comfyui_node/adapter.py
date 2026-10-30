"""potatoforge 适配器的读取与校验。

适配器格式（`potatoforge_moss_encoder_projection_v1`）：:

    weight    fp16 [d_in, 4096]  标准化空间里的线性映射
    mean_in   fp32 [d_in]
    std_in    fp32 [d_in]
    mean_out  fp32 [4096]
    std_out   fp32 [4096]

metadata 里带 `format` / `taps` / `alpha` / `val_cosine` 等，节点据此自描述，
不需要额外传抽头参数。
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from safetensors import safe_open

ADAPTER_FORMAT = "potatoforge_moss_encoder_projection_v1"
REQUIRED_TENSORS = ("weight", "mean_in", "std_in", "mean_out", "std_out")
OUT_DIM = 4096


def parse_taps(text: str) -> list[int]:
    """把 `"2,4,8,12,16,-1"` 解析成下标列表。

    Args:
        text: 逗号分隔的下标，允许空格。

    Returns:
        整数列表，保持输入顺序（顺序决定特征拼接顺序，不能重排）。

    Raises:
        ValueError: 解析不出任何整数。
    """
    taps = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not taps:
        raise ValueError(f"抽头列表为空：{text!r}")
    return taps


def to_comfy_taps(taps: list[int], num_layers: int) -> list[int]:
    """把 transformers 口径的 hidden_states 下标换算成 ComfyUI 的 intermediate_output 下标。

    两边的语义差别：
      transformers：`0` 是 embedding，`k` 是第 k 个 block 的输出，`-1` 是过 final norm 之后；
      ComfyUI：`j` 是第 j 个 block 的**输入**（也就是第 j-1 个 block 的输出），
               `j == num_layers` 才是过 final norm 之后。

    Args:
        taps: transformers 口径的下标。
        num_layers: 模型的 transformer block 数。

    Returns:
        ComfyUI 口径的下标列表，顺序与输入一致。

    Notes:
        正数下标两边相同；负数用 `num_layers + 1 + tap`（-1 → num_layers，-2 → num_layers-1）。
    """
    return [num_layers + 1 + tap if tap < 0 else tap for tap in taps]


def load_adapter(path: str | Path) -> dict:
    """读适配器并校验（按文件 mtime + 大小缓存，命中时不再读盘）。

    Args:
        path: `.safetensors` 路径。

    Returns:
        dict，键为 `weight` / `mean_in` / `std_in` / `mean_out` / `std_out`（均为 CPU 上的
        float32，`weight` 也已升到 float32）+ `meta`（原始 metadata）+ `taps`（下标列表）。
        **返回值是共享缓存**：只能读，不要原地改喵。

    Raises:
        ValueError: 文件不存在、格式串不符、缺张量、形状不匹配、std 非正或含非有限值。
            报错信息直接可读，不用调用方再翻译一遍。

    Notes:
        推理侧按 CPU float32 做投影，显存零占用；权重只有几十 MB，速度不构成瓶颈。
        但整文件读盘 + fp16→fp32 + 逐张量有限性校验约 27 ms，占单次编码的 10%，
        而 ComfyUI 每次执行节点都会重新调用，所以按 (路径, mtime, 大小) 缓存喵。
        改了适配器文件 mtime 就会变，缓存自然失效，不会读到旧权重喵。
    """
    file = Path(str(path).strip().strip('"'))
    try:
        stat = file.stat()
    except OSError as error:
        raise ValueError(f"找不到适配器：{file}") from error
    return _load_adapter_cached(str(file.resolve()), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _load_adapter_cached(resolved: str, mtime_ns: int, size: int) -> dict:
    """真正的读取与校验；`mtime_ns` / `size` 只用来参与缓存键。"""
    file = Path(resolved)
    with safe_open(str(file), framework="pt") as handle:
        meta = handle.metadata() or {}
        missing = [name for name in REQUIRED_TENSORS if name not in handle.keys()]
        if missing:
            raise ValueError(f"{file.name} 缺少张量：{missing}")
        tensors = {name: handle.get_tensor(name).float() for name in REQUIRED_TENSORS}

    fmt = meta.get("format", "")
    if fmt and fmt != ADAPTER_FORMAT:
        raise ValueError(f"适配器格式是 {fmt}，本节点只支持 {ADAPTER_FORMAT}")

    weight = tensors["weight"]
    if weight.ndim != 2 or weight.shape[1] != OUT_DIM:
        raise ValueError(f"weight 形状应为 [d_in, {OUT_DIM}]，实际 {tuple(weight.shape)}")
    d_in = int(weight.shape[0])
    for name in ("mean_in", "std_in"):
        if int(tensors[name].shape[0]) != d_in:
            raise ValueError(f"{name} 长度应为 {d_in}，实际 {int(tensors[name].shape[0])}")
    for name in ("mean_out", "std_out"):
        if int(tensors[name].shape[0]) != OUT_DIM:
            raise ValueError(f"{name} 长度应为 {OUT_DIM}，实际 {int(tensors[name].shape[0])}")
    for name in ("std_in", "std_out"):
        if bool((tensors[name] <= 0).any()):
            raise ValueError(f"{name} 必须为正数（除以它做标准化）")
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 含非有限值")

    return {
        **tensors,
        "sink": None,
        "meta": meta,
        "taps": parse_taps(meta["taps"]) if meta.get("taps") else [-1],
    }


def load_sink(path: str | Path) -> torch.Tensor:
    """读 teacher 的 token 0（attention sink）向量（按 mtime + 大小缓存）。

    Args:
        path: `.npy` 路径，内容为 `[4096]` float32。

    Returns:
        `[4096]` float32 CPU 张量（共享缓存，只读）。

    Raises:
        ValueError: 文件不存在、形状不是 4096 或含非有限值。

    Notes:
        训练时每条 prompt 的 token 0 被剔除，所以投影不覆盖它；
        推理时必须换回 teacher 实测的 sink，否则首位向量范数与方向都对不上喵。
    """
    file = Path(str(path).strip().strip('"'))
    try:
        stat = file.stat()
    except OSError as error:
        raise ValueError(f"找不到 sink 文件：{file}") from error
    return _load_sink_cached(str(file.resolve()), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _load_sink_cached(resolved: str, mtime_ns: int, size: int) -> torch.Tensor:
    """真正的读取与校验；`mtime_ns` / `size` 只用来参与缓存键。"""
    import numpy as np

    file = Path(resolved)
    vector = torch.from_numpy(np.load(str(file))).float().reshape(-1)
    if int(vector.numel()) != OUT_DIM:
        raise ValueError(f"sink 长度应为 {OUT_DIM}，实际 {int(vector.numel())}")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("sink 含非有限值")
    return vector
