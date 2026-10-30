"""teacher 与 student 共用的配置解析与 RoPE 缓冲重建。

两个编码器都从 Qwen3.5 的 VLM 权重里取文本主干，这一步完全相同，
所以只在这里实现一次喵。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import Qwen3_5TextConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

# safetensors 里文本主干的前缀；去掉它就是 transformers 的模块路径。
MODEL_PREFIX = "model.language_model."

# 线性注意力里必须保持 fp32 的参数。官方 config 用 `mamba_ssm_dtype=float32`
# 声明，权重文件里也确实是 F32；统一 cast 成 bf16 会让 exp/softplus 掉精度喵。
SSM_FP32_FIELDS = ("A_log", "dt_bias")


def build_config(config_path: Path | str) -> Qwen3_5TextConfig:
    """从 VLM 的 `config.json` 取出文本部分配置。

    Args:
        config_path: Qwen3.5 系列的 config.json（顶层 `model_type=qwen3_5`）。

    Returns:
        可以直接构建 `Qwen3_5TextModel` 的 `Qwen3_5TextConfig`。

    Notes:
        顶层是 VLM 结构，文本参数在 `text_config` 里喵。直接调
        `Qwen3_5TextConfig.from_pretrained` 会按顶层 model_type 找错类，
        所以手工取子字典；没有 `text_config` 的纯文本 config 原样使用喵。
    """
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return Qwen3_5TextConfig(**raw.get("text_config", raw))


def restore_rope(model: torch.nn.Module) -> None:
    """重建 `persistent=False` 的 RoPE 频率缓冲。

    Args:
        model: 已经装好权重的 `Qwen3_5TextModel`。

    Returns:
        无，原地修改。

    Notes:
        `Qwen3_5TextRotaryEmbedding.inv_freq` 不写进 state_dict，所以从 meta
        构建时它是 meta 张量、前向会直接报错喵。
        初始化时按 `rope_parameters["rope_type"]` 选择初值函数：不是 `default`
        （linear / dynamic / yarn 等）时要用 `ROPE_INIT_FUNCTIONS` 里对应的一项，
        否则在改过 rope 配置的模型上会静默算出错的频率喵。
        频率张量只有 head_dim/2 个 float，留在 CPU 即可，
        forward 里本来就会 `.to(device=x.device)` 喵。
    """
    for module in model.modules():
        if not isinstance(module, Qwen3_5TextRotaryEmbedding):
            continue
        rope_type = getattr(module, "rope_type", "default")
        init_fn = module.compute_default_rope_parameters
        if rope_type != "default":
            if rope_type not in ROPE_INIT_FUNCTIONS:
                raise ValueError(f"不支持的 rope_type {rope_type!r}")
            init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, scaling = init_fn(module.config)
        module.inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
        module.original_inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
        module.attention_scaling = scaling


def restore_ssm_dtype(model: torch.nn.Module) -> None:
    """把线性注意力的 `A_log` / `dt_bias` 恢复成 config 声明的精度。

    Args:
        model: 已经装好权重、可能被整体 cast 过的模型。

    Returns:
        无，原地替换这两个 Parameter。

    Notes:
        前向写的是 `-A_log.float().exp()` 与 `softplus(a.float() + dt_bias)`，
        即假定两者以 fp32 存放。整模型 `.to(dtype=bfloat16)` 会把它们压成 bf16
        （约 3 位十进制有效数字），再 `.float()` 也救不回被舍掉的位喵。
        config 的 `mamba_ssm_dtype` 声明了应有精度，缺失时按 fp32 处理喵。
    """
    target_name = getattr(model.config, "mamba_ssm_dtype", "float32") or "float32"
    target = getattr(torch, target_name, torch.float32)
    for module in model.modules():
        for field in SSM_FP32_FIELDS:
            param = getattr(module, field, None)
            if not isinstance(param, torch.nn.Parameter) or param.dtype == target:
                continue
            module._parameters[field] = torch.nn.Parameter(
                param.data.to(target), requires_grad=param.requires_grad
            )
