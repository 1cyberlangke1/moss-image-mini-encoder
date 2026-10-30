"""9B teacher 文本编码器的加载与前向。

MOSS-Image 用的是 **Qwen3.5-9B 的最后一层隐状态**（`text_encoder.py:35` layer="last"，
不带 lm_head）。这里的加载路径：

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

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)

from .quant import (
    QUANT_SUFFIXES,
    Int8Embedding,
    QuantLinear,
    build_asym_w4a8,
    is_asym_w4a8,
    is_int8_tensorwise,
)

# safetensors 里文本主干的前缀；去掉它就是 transformers 的模块路径。
MODEL_PREFIX = "model.language_model."


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


def _build_config(config_path: Path) -> Qwen3_5TextConfig:
    """从官方 `config.json` 取出文本部分的配置。

    Args:
        config_path: Qwen3.5-9B 的 config.json。

    Returns:
        可以直接构建 `Qwen3_5TextModel` 的配置。

    Notes:
        官方 config 是 VLM 结构（顶层 `model_type=qwen3_5`），文本参数在
        `text_config` 里；`Qwen3_5TextConfig.from_pretrained` 会按顶层
        model_type 找错类，所以这里手工取子字典。
    """
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    text = raw.get("text_config", raw)
    return Qwen3_5TextConfig(**text)


def _restore_rope(model: torch.nn.Module) -> None:
    """重建 meta 构建时丢失的 RoPE 频率缓冲。

    Args:
        model: 已经从 meta 里替换过参数的模型。

    Returns:
        无，原地修改。

    Notes:
        `Qwen3_5TextRotaryEmbedding.inv_freq` 是 `persistent=False` 的 buffer，
        不进 state_dict；meta 构建后它的值不存在，前向会直接报错。
        这里用官方同一个 staticmethod 在 CPU 上重算，形状与训练时一致。
        频率张量很小（head_dim/2 个 float），留在 CPU 即可，
        forward 里本来就会 `.to(device=x.device)`。
    """
    for module in model.modules():
        if isinstance(module, Qwen3_5TextRotaryEmbedding):
            inv_freq, scaling = module.compute_default_rope_parameters(module.config)
            module.inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
            module.original_inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
            module.attention_scaling = scaling


class TeacherEncoder:
    """Qwen3.5-9B 的 w4a8 权重编码器（只出最后一层隐状态）。

    Args:
        ckpt_path: `qwen3.5_9b_w4a8_gptq.safetensors`。
        config_path: 9B 的 config.json。
        device: 计算设备，默认 cuda。
        dtype: 激活与输出 dtype，默认 bfloat16。

    Attributes:
        model: `Qwen3_5TextModel`，全部参数冻结。
        hidden_size: 输出维度（4096）。

    Notes:
        构造完显存占用 = 量化权重 6.84 GiB + int8 词表 1.0 GiB + 激活。
        8 GB 卡上批量前向要自己控制 token 预算。
    """

    def __init__(
        self,
        ckpt_path: Path | str,
        config_path: Path | str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        config = _build_config(Path(config_path))
        config._attn_implementation = "sdpa"

        with torch.device("meta"):
            model = Qwen3_5TextModel(config)

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
            if verbose:
                print(
                    f"[teacher] 量化层 {len(quant_prefixes)} 个，int8 查表层 {len(int8_prefixes)} 个",
                    flush=True,
                )

            self._load_plain(model, handle, keys, quant_prefixes | int8_prefixes)
            self._swap_quantized(model, _TensorView(handle), quant_prefixes, int8_prefixes)

        _restore_rope(model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.hidden_size = int(config.hidden_size)

    def _load_plain(
        self,
        model: torch.nn.Module,
        handle,
        keys: set[str],
        quant_prefixes: set[str],
    ) -> None:
        """把非量化张量装进 meta 模型。

        Args:
            model: meta 上构建的模型。
            handle: 打开的 safetensors。
            keys: 全部 key。
            quant_prefixes: 量化层前缀（它们的 `.weight` 不是浮点张量，要排除）。

        Returns:
            无。

        Notes:
            用 `assign=True` 才会替换 meta 参数而不是往 meta 里 copy。
            `strict=False` 是必须的：量化层的 weight 不在 state_dict 里，
            稍后由 `_swap_quantized` 替换。dtype 保持文件原样
            （`A_log` 是 f32，转 bf16 会让 `exp(A_log)` 掉精度）。
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
            state[name] = handle.get_tensor(key).to(self.device)

        model.load_state_dict(state, strict=False, assign=True)
        del state

    def _swap_quantized(
        self,
        model: torch.nn.Module,
        handle,
        quant_prefixes: set[str],
        int8_prefixes: set[str],
    ) -> None:
        """把量化权重换成 `QuantLinear` / `Int8Embedding`。

        Args:
            model: 已装好普通参数的模型。
            handle: 打开的 safetensors。
            quant_prefixes: asym_w4a8 层前缀（safetensors 命名）。
            int8_prefixes: int8_tensorwise 层前缀。

        Returns:
            无，原地替换子模块。

        Notes:
            bias 从刚 assign 进来的原模块上取，替换后仍指向同一张量。
            量化层在 safetensors 里没有 bias 就传 None。
        """
        for prefix in quant_prefixes:
            name = prefix[len(MODEL_PREFIX) :]
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            old = getattr(parent, leaf)
            bias = getattr(old, "bias", None)
            weight = build_asym_w4a8(prefix, handle, self.device, self.dtype)
            setattr(parent, leaf, QuantLinear(weight, bias))

        for prefix in int8_prefixes:
            name = prefix[len(MODEL_PREFIX) :]
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            weight = handle[f"{prefix}.weight"].to(self.device)
            scale = handle[f"{prefix}.weight_scale"].to(self.device)
            setattr(parent, leaf, Int8Embedding(weight, scale, self.dtype))

    @torch.inference_mode()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """跑一次前向，返回最后一层隐状态。

        Args:
            input_ids: `[B, T]` int64，右侧 padding，且 pad 值必须是 248044。
            attention_mask: `[B, T]`，1 表示真实 token。None 表示全是真实 token。

        Returns:
            `[B, T, 4096]` bf16。padding 位置的值无意义，调用方按 mask 取。

        Notes:
            右 padding + causal mask 下，真实 token 不会看到后面的 pad，
            所以只要按 mask 截取，结果与逐条编码一致。
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state
