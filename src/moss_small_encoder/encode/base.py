"""编码器的公共前向：补齐与变长两种走法只有这一份实现喵。

输入: 已装载好的 `Qwen3_5TextModel` 喵。
输出: `_EncoderBase`，供教师与学生两个门面类继承喵。
预期: 补齐走 sdpa、变长走 flash varlen，两处逻辑都只写一次喵。
"""

from __future__ import annotations

import torch


class _EncoderBase:
    """所有编码器共用的前向实现喵。

    属性:
        model: 已冻结的 `Qwen3_5TextModel` 喵。
        hidden_size: 输出隐层维度喵。
    """

    model: torch.nn.Module
    hidden_size: int

    @torch.inference_mode()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        taps: tuple[int, ...] = (-1,),
    ) -> dict[int, torch.Tensor]:
        """跑一次补齐前向，返回指定层的隐状态喵。

        输入: `input_ids` 形状 `[B, T]`；`attention_mask` 形状 `[B, T]` 或 None喵。
        输出: 层下标到 `[B, T, hidden_size]` 的映射喵。
        预期: 一次前向同时取全部 tap，避免重复计算喵。
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        states = out.hidden_states
        if states is None:
            raise RuntimeError("模型没有返回 hidden_states 喵")
        return {tap: states[tap] for tap in taps}

    @torch.inference_mode()
    def encode_packed(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        taps: tuple[int, ...] = (-1,),
    ) -> dict[int, torch.Tensor]:
        """跑一次变长前向，多条序列首尾相接喵。

        输入: `input_ids` 形状 `[1, total]`；`cu_seqlens` 形状 `[N+1]`喵。
        输出: 层下标到 `[1, total, hidden_size]` 的映射喵。
        预期: 段边界由 cu_seqlens 决定，不传 attention_mask喵。
        """
        from ..packing import packed_forward_kwargs

        out = self.model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            **packed_forward_kwargs(cu_seqlens),
        )
        states = out.hidden_states
        if states is None:
            raise RuntimeError("模型没有返回 hidden_states 喵")
        return {tap: states[tap] for tap in taps}
