"""ComfyUI 节点共用的适配器读取与校验。

适配器格式（`potatoforge_moss_encoder_projection_v1`）：:

    weight    fp16 [d_in, 4096]  标准化空间里的线性映射
    mean_in   fp32 [d_in]
    std_in    fp32 [d_in]
    mean_out  fp32 [4096]
    std_out   fp32 [4096]
    sink      fp32 [4096]        teacher 的 token 0 向量

metadata 里带 `format` / `taps` / `alpha` / `val_cosine` 等，节点据此自描述。
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from safetensors import safe_open

ADAPTER_FORMAT = "potatoforge_moss_encoder_projection_v1"
REQUIRED_TENSORS = ("weight", "mean_in", "std_in", "mean_out", "std_out")
# sink 可选：新产物写在文件里，老产物没有就跳过 token 0 替换喵
OPTIONAL_TENSORS = ("sink",)
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
        dict，六个张量（`weight` 升到 float32，其余本就是 fp32）+ `meta` + `taps`。
        **返回值是共享缓存**：只能读，不要原地改喵。

    Raises:
        ValueError: 文件不存在、格式串不符、缺张量、形状不匹配、std 非正或含非有限值。

    Notes:
        整文件读盘 + fp16→fp32 + 逐张量有限性校验约 27 ms，占单次编码的 10%，
        而 ComfyUI 每次执行节点都会重新调用，所以按 (路径, mtime, 大小) 缓存喵。
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
        keys = tuple(name for name in (*REQUIRED_TENSORS, *OPTIONAL_TENSORS) if name in handle.keys())
        tensors = {name: handle.get_tensor(name).float() for name in keys}

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
        if int(tensors[name].numel()) != OUT_DIM:
            raise ValueError(f"{name} 长度应为 {OUT_DIM}，实际 {int(tensors[name].numel())}")
    if "sink" in tensors and int(tensors["sink"].numel()) != OUT_DIM:
        raise ValueError(f"sink 长度应为 {OUT_DIM}，实际 {int(tensors['sink'].numel())}")
    for name in ("std_in", "std_out"):
        if bool((tensors[name] <= 0).any()):
            raise ValueError(f"{name} 必须为正数（除以它做标准化）")
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} 含非有限值")

    if "sink" in tensors:
        tensors["sink"] = tensors["sink"].reshape(-1)
    return {
        **tensors,
        "key": (resolved, mtime_ns, size),
        "meta": meta,
        "taps": parse_taps(meta["taps"]) if meta.get("taps") else [-1],
    }


_DEVICE_CACHE: dict[tuple, dict] = {}


def adapter_on(loaded: dict, device: torch.device) -> dict:
    """把适配器张量放到指定设备（按「适配器身份 + 设备」缓存）。

    Args:
        loaded: `load_adapter` 的返回值（CPU float32）。
        device: 目标设备，通常是 clip 输出所在的设备。

    Returns:
        dict，键同 `load_adapter`，但张量都已搬到 `device`。共享缓存，只读喵。

    Notes:
        权重约 48 MB，每次执行都 `.to(cuda)` 要 8 ms，正好把 GPU 投影省的
        7 ms 吃回去；所以第一次搬完就缓存，之后直接复用喵。
        CPU 上调用时返回的就是原张量，不做任何复制喵。
    """
    if device.type == "cpu":
        return loaded
    key = (loaded["key"], str(device))
    hit = _DEVICE_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_DEVICE_CACHE) > 15:
        # 只有换适配器文件或换设备才会进新条目，正常使用不会涨到 16 条喵
        _DEVICE_CACHE.clear()
    moved = {
        name: loaded[name].to(device)
        for name in (*REQUIRED_TENSORS, *OPTIONAL_TENSORS)
        if name in loaded
    }
    _DEVICE_CACHE[key] = moved
    return moved
