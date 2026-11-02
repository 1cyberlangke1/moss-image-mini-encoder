"""统一的文本编码器：教师（单文件量化权重）与学生（目录权重）共用。

MOSS-Image 用的是 **Qwen3.5-9B 的最后一层隐状态**（`text_encoder.py:35` layer="last"，
不带 lm_head）。学生侧要记的是同一套隐状态，只是维度不同。

量化加载路径：

1. 用 `transformers` 的 `Qwen3_5TextModel` 在 meta 设备上搭结构（真权重 18.4 GB
   bf16，8 GB 显存放不下，也不能先在 CPU 上实体化）；
2. 非量化张量（norm / conv1d / A_log / dt_bias / 小投影）直接 `assign=True` 装进 meta
   参数；
3. 量化张量交给 `comfy_kitchen` 的 `QuantizedTensor`，权重全程 int8 常驻显存
   （6.84 GiB），矩阵乘走融合 kernel。

跳过的部分：`mtp.*`（投机解码头，BF16）、`model.visual.*`（视觉塔）、`lm_head`
（MOSS 不使用），它们合计约 4 GB。
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from transformers.utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

from .common import MODEL_PREFIX, build_config, restore_rope, restore_ssm_dtype
from ..packing import packed_forward_kwargs
from .base import _EncoderBase

from .quant import (
    build_int8_tensorwise,
    QUANT_SUFFIXES,
    Int8Embedding,
    QuantLinear,
    build_asym_w4a8,
    is_asym_w4a8,
    is_int8_tensorwise,
    read_quant_blob,
)

class _TensorView:
    """把 `safe_open` 包成只读映射，让 `build_asym_w4a8` 能按 key 取值。

    Args:
        handle: `safetensors.safe_open` 返回的对象。

    Notes:
        `safe_open` 只提供 `get_tensor`/`keys`，不支持 `in` 和 `[]`；
        量化层要按「有没有某个 scale 张量」来判断布局，所以需要这层适配。
        key 集合只在构造时取一次，避免每个量化层都重新解析 header。
    """

    def __init__(self, handle) -> None:
        self._handle = handle
        self._keys = set(handle.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._handle.get_tensor(key)

    def get(self, key: str, default=None):
        return self._handle.get_tensor(key) if key in self._keys else default


def load_plain_tensors(model, handle, keys, quant_prefixes, device) -> None:
    """把非量化张量装进 meta 模型。

    Args:
        model: meta 上构建的模型。
        handle: 打开的 safetensors。
        keys: 全部 key。
        quant_prefixes: 量化层前缀（它们的 `.weight` 不是浮点张量，要排除）。
        device: 目标设备。

    Returns:
        无。

    Notes:
        用 `assign=True` 才会替换 meta 参数而不是往 meta 里 copy喵。
        `strict=False` 是必须的：量化层的 weight 不在 state_dict 里，
        稍后由 `swap_quantized_modules` 替换喵。但 missing 必须**逐个核对**，
        否则文件里少了一批非量化张量也照样返回，模型会带着 meta 张量跑到
        前向才炸（甚至得到垃圾结果）喵。dtype 保持文件原样
        （`A_log` 是 f32，转 bf16 会让 `exp(A_log)` 掉精度）喵。
    """
    skip = {f"{prefix}.weight" for prefix in quant_prefixes}
    state = {}
    for key in keys:
        if not key.startswith(MODEL_PREFIX):
            continue
        if key in skip:
            continue
        if any(key.endswith("." + suffix) for suffix in QUANT_SUFFIXES):
            continue
        name = key[len(MODEL_PREFIX) :]
        state[name] = handle.get_tensor(key).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    # 允许缺的只有量化层的 weight（下面换成 QuantLinear / Int8Embedding）喵
    expected_missing = {key[len(MODEL_PREFIX) :] for key in skip}
    real_missing = sorted(set(missing) - expected_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            f"权重不匹配：缺 {real_missing[:5]}（共 {len(real_missing)}），"
            f"多 {sorted(unexpected)[:5]}"
        )


def swap_quantized_modules(
    model,
    handle,
    quant_prefixes,
    int8_prefixes,
    device,
    dtype,
) -> None:
    """把量化权重换成 `QuantLinear` / `Int8Embedding`。

    Args:
        model: 已装好普通参数的模型。
        handle: 打开的 safetensors。
        quant_prefixes: asym_w4a8 层前缀（safetensors 命名）。
        int8_prefixes: int8_tensorwise 层前缀。
        device: 目标设备。
        dtype: 计算 dtype。

    Returns:
        无，原地替换子模块。

    Notes:
        bias 从刚 assign 进来的原模块上取，替换后仍指向同一张量喵。
        量化层在 safetensors 里没有 bias 就传 None喵。
    """
    for prefix in quant_prefixes:
        name = prefix[len(MODEL_PREFIX) :]
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, leaf)
        bias = getattr(old, "bias", None)
        weight = build_asym_w4a8(prefix, handle, device, dtype)
        setattr(parent, leaf, QuantLinear(weight, bias))

    for prefix in int8_prefixes:
        name = prefix[len(MODEL_PREFIX) :]
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, leaf)
        if isinstance(old, torch.nn.Embedding):
            # 词表：按 id 查表后仅反量化命中的行
            weight = handle[f"{prefix}.weight"].to(device)
            scale = handle[f"{prefix}.weight_scale"].to(device)
            config = {}
            if f"{prefix}.comfy_quant" in handle:
                config = read_quant_blob(handle[f"{prefix}.comfy_quant"])
            setattr(
                parent,
                leaf,
                Int8Embedding(
                    weight,
                    scale,
                    dtype,
                    convrot=bool(config.get("convrot", False)),
                    convrot_groupsize=int(config.get("convrot_groupsize", 256)),
                ),
            )
        else:
            bias = getattr(old, "bias", None)
            # int8_tensorwise 的 Linear 一律反量化成稠密权重，不保留量化算子路径。
            # 理由：int8 的唯一价值是省体积，不省算力；本机上走 comfy_kitchen 的
            # int8_linear 融合算子只有 110 token/s（同批形状下其它路径 4000+），而
            # 0.8B / 2B 反量化后只占 1.6 / 4.5 GB，8 GB 卡放得下。
            # 9B 那份文件里 int8 只出现在 embed_tokens（Embedding），不经过这里。
            quant = build_int8_tensorwise(prefix, handle, device, dtype)
            weight = quant.dequantize().to(dtype)
            dense = torch.nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                dense.weight.copy_(weight)
                if bias is not None:
                    dense.bias.copy_(bias.to(dtype))
            setattr(parent, leaf, dense)


def _require_packed_deps() -> None:
    """检查 packed 需要的三项能力，缺任何一项立刻失败。

    Raises:
        ImportError: 缺 `flash-attn` / `causal-conv1d` / `flash-linear-attention` 之一。

    Notes:
        三项都不能退化成别的实现：
        缺 `flash-attn` 就没有 varlen 接口，`sdpa` 会忽略 `cu_seqlens`；
        缺 `flash-linear-attention`，gated delta rule 退回 PyTorch 版，不认 `cu_seqlens`；
        缺 `causal-conv1d`，卷积退回 `F.conv1d`，而它不看 `seq_idx`。
        后两种退化都是静默的跨段串味，结果看起来正常但数值全错，所以这里直接拦住。
    """
    missing: list[str] = []
    try:
        import flash_attn  # noqa: F401
    except Exception:
        missing.append("flash-attn")
    if not is_causal_conv1d_available():
        missing.append("causal-conv1d")
    if not is_flash_linear_attention_available():
        missing.append("flash-linear-attention")
    if missing:
        raise ImportError(
            "packed 模式需要 " + " / ".join(missing) + "，安装方法见 docs/pipeline.md"
        )


class TextEncoder(_EncoderBase):
    """统一的文本编码器：目录权重或单文件量化权重，补齐与变长两种前向喵。

    输入: 模型目录（含 config.json 与分片）或单个 safetensors 加 config 路径喵。
    输出: 具备 `encode` / `encode_packed` 的编码器，隐层维度见 `hidden_size`喵。
    预期: 两种装载方式只保留一份实现，前向逻辑全部来自 `_EncoderBase`喵。
    """

    def __init__(
        self,
        source: Path | str,
        config_path: Path | str | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = False,
        packed: bool = False,
        quant_path: Path | str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.packed = packed
        if packed:
            _require_packed_deps()

        if config_path is None:
            model_dir = Path(source)
            config = build_config(model_dir / "config.json")
            config._attn_implementation = "flash_attention_2" if packed else "sdpa"
            if quant_path is None:
                model = Qwen3_5TextModel(config)
                shards = sorted(model_dir.glob("model*.safetensors"))
                if not shards:
                    raise FileNotFoundError(f"{model_dir} 下没有 safetensors")
                state: dict[str, torch.Tensor] = {}
                for shard in shards:
                    with safe_open(str(shard), framework="pt", device="cpu") as handle:
                        for key in handle.keys():
                            if key.startswith(MODEL_PREFIX):
                                state[key[len(MODEL_PREFIX):]] = handle.get_tensor(key)
                missing, unexpected = model.load_state_dict(state, strict=False)
                del state
                real_missing = [n for n in missing if "inv_freq" not in n]
                if real_missing or unexpected:
                    raise RuntimeError(
                        f"权重不匹配：缺 {real_missing[:5]}（共 {len(real_missing)}），"
                        f"多 {list(unexpected)[:5]}"
                    )
                restore_rope(model)
                model.to(self.device, dtype=dtype)
                restore_ssm_dtype(model)
            else:
                self._load_quantized(model, Path(quant_path))
        else:
            config = build_config(Path(config_path))
            config._attn_implementation = "flash_attention_2" if packed else "sdpa"
            with torch.device("meta"):
                model = Qwen3_5TextModel(config)
            self._load_quantized(model, Path(source))

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.hidden_size = int(config.hidden_size)

    def _load_quantized(self, model, ckpt_path: Path) -> None:
        """按 comfy 的量化格式装载权重喵。

        输入: 空的（meta）模型与 safetensors 路径喵。
        输出: 无返回值，模型被就地填充喵。
        预期: 量化层走 comfy_kitchen，其余走普通张量，词表走 int8 反量化喵。
        """
        from .quant import is_asym_w4a8

        with safe_open(str(ckpt_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            quant_prefixes = {
                key[: -len(".weight_s_rel")]
                for key in keys
                if key.endswith(".weight_s_rel") and key.startswith(MODEL_PREFIX)
            }
            int8_prefixes = {
                key[: -len(".weight_scale")]
                for key in keys
                if key.endswith(".weight_scale")
                and key.startswith(MODEL_PREFIX)
                and not is_asym_w4a8(key[: -len(".weight_scale")], keys)
            }
            load_plain_tensors(
                model, handle, keys, quant_prefixes | int8_prefixes, self.device
            )
            swap_quantized_modules(
                model,
                _TensorView(handle),
                quant_prefixes,
                int8_prefixes,
                self.device,
                self.dtype,
            )
        restore_rope(model)
        restore_ssm_dtype(model)
