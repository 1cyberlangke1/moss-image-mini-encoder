"""0.8B student 文本编码器的加载与前向。

student 是 Qwen3.5-0.8B（VLM 结构，文本主干在 `model.language_model`）。
它 1.6 GB bf16，8 GB 卡装得下，所以不需要量化，直接用 transformers 建结构再灌权重。

关键约定：

- **tokenizer 与 teacher 完全相同**（同一份 tokenizer.json），因此 token id 与位置
  天然逐位对齐，投影才能按 token 配对；
- **tap 用 `output_hidden_states` 取**：`hidden_states[0]` 是 embedding 输出，
  `hidden_states[k]` 是第 k 层输出，`hidden_states[-1]` 已经过 final norm
  （对应参考项目的 `last_normalized` 语义）；
- 0.8B 是混合注意力：24 层里 18 层 linear_attention + 6 层 full_attention，
  `config.layer_types` 决定每层类型。
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from .common import MODEL_PREFIX, build_config, restore_rope, restore_ssm_dtype


class StudentEncoder:
    """Qwen3.5-0.8B 编码器，可一次取出多层隐状态。

    Args:
        model_dir: 含 `config.json` 与 safetensors 的目录。
        device: 计算设备。
        dtype: 输出 dtype。
        quant_path: 量化权重文件（`comfy_kitchen` 格式）。给定时走量化装载路径，
            transformers 装不了这类张量，必须复用 teacher 的 `QuantLinear` 路径喵。
        attn: 注意力实现名。默认 `sdpa`；传 `flash_attention_2` 才能让变长打包生效，
            因为 sdpa 不认 `cu_seq_lens`，会把整批当成一条序列互相看见喵。

    Attributes:
        model: `Qwen3_5TextModel`，全部参数冻结。
        hidden_size: 隐状态维度（1024）。
        n_layers: 层数（24）。

    Notes:
        显存占用约 1.7 GB（权重）+ 激活。允许和 teacher 同时驻留：
        6.84 + 1.0 + 1.7 ≈ 9.5 GB，超过 8 GB 物理显存，
        所以编码时 teacher 与 student **必须分两次跑**，不能同时加载。
    """

    def __init__(
        self,
        model_dir: Path | str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quant_path: Path | str | None = None,
        attn: str = "sdpa",
    ) -> None:
        model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        config = build_config(model_dir / "config.json")
        config._attn_implementation = attn

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
                            state[key[len(MODEL_PREFIX) :]] = handle.get_tensor(key)

            missing, unexpected = model.load_state_dict(state, strict=False)
            del state
            # 只允许缺 RoPE 缓冲（persistent=False）；其余缺失说明权重与结构对不上。
            real_missing = [name for name in missing if "inv_freq" not in name]
            if real_missing or unexpected:
                raise RuntimeError(
                    f"student 权重不匹配：缺 {real_missing[:5]}（共 {len(real_missing)}），"
                    f"多 {list(unexpected)[:5]}"
                )

            restore_rope(model)
            model.to(self.device, dtype=dtype)
            # `.to(dtype)` 会把 A_log / dt_bias 一起压成 bf16，这两个必须保持 fp32 喵
            restore_ssm_dtype(model)
        else:
            # 量化权重：复用 teacher 的装载路径，不抄第二份实现喵。
            # 注意换完之后不能再 `.to(device, dtype)`：量化层是自定义张量，`.to` 不认它喵。
            from .quant import is_asym_w4a8
            from .teacher import _TensorView, load_plain_tensors, swap_quantized_modules

            with torch.device("meta"):
                model = Qwen3_5TextModel(config)
            with safe_open(str(quant_path), framework="pt", device="cpu") as handle:
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

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        self.hidden_size = int(config.hidden_size)
        self.n_layers = int(config.num_hidden_layers)

    @torch.inference_mode()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        taps: tuple[int, ...] = (-1,),
    ) -> dict[int, torch.Tensor]:
        """一次前向，取出若干个 tap 的隐状态。

        Args:
            input_ids: `[B, T]` int64。
            attention_mask: `[B, T]` 或 None。
            taps: 层下标，`-1` 表示最后一层（已过 final norm）。
                取值 `[-1, n_layers]`，其中 `0` 是 embedding 输出。

        Returns:
            层下标到 `[B, T, hidden_size]` 的映射，dtype 与构造时一致。

        Notes:
            单次前向同时取出所有 tap，避免多层扫描时重复计算。
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("student 没有返回 hidden_states")
        return {tap: hidden_states[tap] for tap in taps}
