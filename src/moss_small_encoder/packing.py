"""变长打包的公共工具喵。

打包的前向需要三样东西：把多条序列首尾相接的 `cu_seqlens`、
按段从 0 重开的 `position_ids`、以及在段边界断开的 `seq_idx`喵。

`position_ids` 与 `seq_idx` 都只由 `cu_seqlens` 决定，所以统一从这里生成，
避免教师侧与训练侧各写一份、改一处忘一处喵。
"""

from __future__ import annotations

import torch


def packing_metadata(
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由累积长度造出 `position_ids` 与 `seq_idx`喵。

    输入: `cu_seqlens`，`[N + 1]` 整型，单调递增且首项为 0
    输出: `(position_ids, seq_idx)`
          `position_ids` 是 `[4, 1, total]`，四条 mrope 通道取同一份值；
          `seq_idx` 是 `[1, total]` int32，第 i 段全为 i
    预期: `position_ids[0, 0]` 在每段起点回到 0，`seq_idx` 在段边界递增

    Notes:
        位置必须按段重置，否则第 2 条起会带上前一条的位置偏移；
        `seq_idx` 让卷积核在段边界重置状态，否则前一条的尾部会卷进后一条喵。
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    device = cu_seqlens.device
    positions = torch.cat(
        [torch.arange(length, device=device, dtype=torch.long) for length in lengths]
    )
    position_ids = positions.view(1, 1, -1).expand(4, 1, -1)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32),
        torch.tensor(lengths, device=device),
    ).unsqueeze(0)
    return position_ids, seq_idx


def packed_forward_kwargs(cu_seqlens: torch.Tensor) -> dict:
    """给模型前向准备打包用的关键字喵。

    输入: `cu_seqlens`，`[N + 1]`
    输出: 可以直接 `**` 展开的字典，含 `cu_seq_lens_q`、`cu_seq_lens_k`、
          `max_length_q`、`max_length_k`、`seq_idx`、`position_ids` 与 `attention_mask`
    预期: `attention_mask` 传字典是为了跳过掩码构造，varlen 分支要求它是 None

    Notes:
        这几个键不是每个算子都认，`transformers` 的包装器会按目标函数的
        参数表过滤，所以可以一次性全部传下去喵。
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    position_ids, seq_idx = packing_metadata(cu_seqlens)
    longest = int(max(lengths))
    return {
        "attention_mask": {"full_attention": None, "linear_attention": None},
        "position_ids": position_ids,
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": longest,
        "max_length_k": longest,
        "seq_idx": seq_idx,
    }
