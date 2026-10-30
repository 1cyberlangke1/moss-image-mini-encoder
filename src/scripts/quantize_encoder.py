"""把 bf16 学生编码器量化成 ComfyUI 能直接 `CLIPLoader` 的格式。

输入:
    学生模型目录（含 `model.safetensors*` 与 `config.json`）
输出:
    单个 `.safetensors`，量化层写成 `weight` / `weight_s_rel` / `weight_s_channel` /
    `weight_codebook` / `comfy_quant`，其余张量原样保留。
预期:
    同一次量化是纯权重量化（RTN + Hadamard），不需要任何校准数据，CPU/GPU 都能跑完。

量化哪些层，按 Qwen 官方 FP8 的 `modules_to_not_convert` 清单喵：
    - `linear_attn.conv1d` / `in_proj_a` / `in_proj_b`（卷积核与门控 logits，量化必崩）
    - `embed_tokens` / `lm_head`（词表查表）
    - `visual.*`（视觉塔，本任务不用）
其余 Linear（q/k/v/o、gate/up/down、in_proj_qkvz、out_proj）全部量化喵。

用法:
    python scripts/quantize_encoder.py --src models/qwen3.5-2b \\
        --format w4a8 --out models/qwen3.5-2b-w4a8.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# 官方 FP8 的排除清单（去掉 MoE 的 mlp.gate，稠密模型没有它喵）
EXCLUDE_PATTERNS = (
    ".linear_attn.conv1d",
    ".linear_attn.in_proj_a.",
    ".linear_attn.in_proj_b.",
    ".embed_tokens.",
    "lm_head.",
    ".visual.",
)


def format_conf(fmt: str) -> dict:
    """量化层要写进 comfy_quant 的 JSON。

    Args:
        fmt: `int8` 或 `w4a8`。

    Returns:
        可直接 JSON 序列化的 dict，字段名与 ComfyUI 的 loader 对齐。
    """
    if fmt == "w4a8":
        return {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
    return {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def load_state_dict(model_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """读入完整权重与 metadata。

    Args:
        model_dir: 学生模型目录，支持单文件或 `*.index.json` 分片。

    Returns:
        `(张量字典, metadata)`，张量仍在 CPU。

    Raises:
        FileNotFoundError: 目录里没有 safetensors。
    """
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        files = sorted({model_dir / name for name in json.loads(index.read_text())["weight_map"].values()})
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{model_dir} 里没有 safetensors")

    tensors: dict[str, torch.Tensor] = {}
    meta: dict = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            meta.update(handle.metadata() or {})
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors, meta


def should_quantize(key: str, weight: torch.Tensor, fmt: str) -> bool:
    """判断某个 `*.weight` 是否该量化。

    Args:
        key: 张量名。
        weight: 对应权重。
        fmt: `int8` 或 `w4a8`。

    Returns:
        True 表示量化。

    Notes:
        只处理二维权重（Linear）；卷积核、norm、词表、视觉塔按官方清单排除喵。
        K 轴必须能被 Hadamard 组宽整除，否则量化核会拒绝，直接跳过以免整个转换失败喵。
    """
    if not key.endswith(".weight") or weight.ndim != 2:
        return False
    if any(pattern in f".{key}" for pattern in EXCLUDE_PATTERNS):
        return False
    in_features = int(weight.shape[1])
    need = 256 if fmt == "w4a8" else 256
    return in_features % need == 0 and in_features % 16 == 0


def quantize_one(weight: torch.Tensor, fmt: str, device: torch.device):
    """量化单个权重矩阵。

    Args:
        weight: `[N, K]` 浮点权重。
        fmt: `int8` 或 `w4a8`。
        device: 计算设备。

    Returns:
        `(QuantizedTensor, 键前缀后缀字典)`，键名已按 ComfyUI 约定生成。

    Notes:
        量化器来自 `comfy_kitchen`，与 ComfyUI 生产路径完全同一份实现喵——
        w4a8 是码本 + 分组 fp8 scale + Hadamard，int8 是逐通道 scale + Hadamard喵。
    """
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

    source = weight.to(device=device, dtype=torch.float16)
    if fmt == "w4a8":
        qdata, params = AsymW4A8Int8Layout.quantize(source)
        layout = "AsymW4A8Int8Layout"
    else:
        qdata, params = TensorWiseINT8Layout.quantize(
            source, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256
        )
        layout = "TensorWiseINT8Layout"
    return QuantizedTensor(qdata, layout, params), params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--format", choices=("int8", "w4a8"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个可量化层（调试）")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensors, meta = load_state_dict(args.src)
    print(f"读入 {len(tensors)} 个张量 | 设备 {device} | 格式 {args.format}", flush=True)

    conf = format_conf(args.format)
    blob = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)

    out: dict[str, torch.Tensor] = {}
    quantized = 0
    skipped = 0
    for key, value in tensors.items():
        if not should_quantize(key, value, args.format):
            out[key] = value
            if key.endswith(".weight"):
                skipped += 1
            continue
        prefix = key[: -len(".weight")]
        quantized_tensor, _params = quantize_one(value, args.format, device)
        for name, tensor in quantized_tensor.state_dict(prefix + ".weight").items():
            out[name] = tensor.contiguous().cpu()
        out[prefix + ".comfy_quant"] = blob.clone()
        quantized += 1
        if args.limit is not None and quantized >= args.limit:
            break
        if quantized % 40 == 0:
            print(f"  已量化 {quantized} 层", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["moss_quant"] = json.dumps(conf)
    save_file(out, str(args.out), metadata=meta)

    size_mb = args.out.stat().st_size / 1024**2
    print(f"完成：量化 {quantized} 层，跳过 {skipped} 层权重 → {args.out} ({size_mb:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
