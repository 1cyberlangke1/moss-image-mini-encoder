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
        packed: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.packed = packed
        if packed:
            _require_packed_deps()
        config = build_config(Path(config_path))
        # packed 走 flash_attention_2：只有它的 varlen 接口认 cu_seqlens，
        # sdpa 会静默忽略这些键，结果就是跨段串味而不报错。
        config._attn_implementation = "flash_attention_2" if packed else "sdpa"

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

        restore_rope(model)
        restore_ssm_dtype(model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.hidden_size = int(config.hidden_size)

    def _load_plain(self, model, handle, keys, quant_prefixes) -> None:
        """见模块级 `load_plain_tensors`。"""
        load_plain_tensors(model, handle, keys, quant_prefixes, self.device)

    def _swap_quantized(self, model, handle, quant_prefixes, int8_prefixes) -> None:
        """见模块级 `swap_quantized_modules`。"""
        swap_quantized_modules(
            model, handle, quant_prefixes, int8_prefixes, self.device, self.dtype
        )

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

    @torch.inference_mode()
    def encode_packed(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """把多条序列首尾相接，一次前向，返回拼平后的隐状态。

        Args:
            input_ids: `[1, total]` int64，多条 prompt 的 token 直接连在一起，中间不插 pad。
            cu_seqlens: `[N + 1]` int32，累积长度，单调递增且首项为 0、末项为 total。

        Returns:
            `[1, total, hidden]` bf16。第 i 条序列对应 `cu_seqlens[i] : cu_seqlens[i + 1]`。

        Notes:
            `position_ids` 与 `seq_idx` 都从 `cu_seqlens` 反推，不额外传参，避免两边不一致。
            `position_ids` 按段从 0 重开，否则第 2 条起会带上前一条的位置偏移。
            `seq_idx` 让 kernel 等于 4 的 causal conv1d 在段边界重置，
            否则前一条的尾部会卷进后一条的开头。
            `attention_mask` 传字典是为了跳过 mask 构造：varlen 分支要求它是 None。
        """
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        out = self.model(
            input_ids=input_ids,
            use_cache=False,
            **packing_forward_kwargs(cu_seqlens),
        )
        return out.last_hidden_state
