"""微调 0.8B 主干的训练循环：LoRA + CKA + 线性头联合优化喵。

解决的问题
----------
冻结主干、只用 Ridge 求闭式解时，投影只能搬运学生已有的方向。
若某个「专名→外观」的绑定压根不在 0.8B 的隐状态里，线性映射造不出来，
表现为加入大量角色名样本后出图仍然认错人。

本模块把主干打开（LoRA），直接以教师的隐状态为标签训练，
先回答一个前置问题：0.8B 的权重到底能不能装下这个绑定。

损失
----
`L = w_cka·(1−CKA) + w_cos·(1−cos(P·h, y)) + w_col·塌缩正则`

- CKA：只看 token 之间的相对结构，对旋转、缩放、维度都不敏感，
  学生 6144 维与教师 4096 维可以直接比；
- 余弦：经可训练线性头 P 投到教师空间后逐 token 对齐，用来钉住基空间；
- 塌缩正则：批内 token 两两余弦的平方均值，防止所有 token 收敛到同一方向。

训练完必须重解一次 Ridge：LoRA 改了主干，旧闭式解全部失效喵。

用法
----
    python -m moss_small_encoder.train.finetune \
        --features dataset/features/v2 --adapter output/qwen3.5-0.8b/projection_linear.safetensors \
        --out output/qwen3.5-0.8b-finetuned --samples 20 --steps 200

过拟合实验用 `--samples N` 只取前 N 条角色名样本，训练损失能压到接近 0
就说明容量够，压不下去就说明该绑定不在这个学生里喵。
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .cka import cka_loss, pairwise_cosine_penalty
from ..packing import packed_forward_kwargs

# 与 encode 阶段一致的抽头层喵。
TAPS_DEFAULT = (2, 4, 8, 12, 16, -1)

# 150 层命中：两个投影、MLP 三个、线性注意力的 in_proj 三个喵。
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "out_proj",
    "in_proj_qkvz", "in_proj_a", "in_proj_b",
]

PAD_ID = 248044


@dataclass
class FinetuneConfig:
    """微调超参数喵。

    字段默认值取自参考实现与本次容量实验：
    LoRA r=16 / alpha=32 / dropout=0.05 可训练参数约占 1.04%；
    线性头学习率比 LoRA 低一个量级，避免一开始就把 Ridge 解带偏喵。
    """

    taps: tuple[int, ...] = TAPS_DEFAULT
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr_lora: float = 1e-3
    steps: int = 200
    batch_tokens: int = 4096
    packed: bool = False
    pad_tiers: tuple[int, ...] | None = None
    eval_every: int = 0
    cka_weight: float = 1.0
    cka_bf16: bool = True
    collapse_weight: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 50
    keep_last: int = 0
    min_lr_ratio: float = 0.01
    seed: int = 20261027


@dataclass
class FinetuneResult:
    """训练结果喵。

    Attributes:
        steps: 实际完成的步数。
        history: 每 `log_every` 步一条记录，含 loss 与分项。
    """

    steps: int
    history: list[dict] = field(default_factory=list)

    @property
    def first(self) -> dict:
        """首条记录喵。"""
        return self.history[0] if self.history else {}

    @property
    def last(self) -> dict:
        """末条记录喵。"""
        return self.history[-1] if self.history else {}


def decoder_layers(model):
    """取出 PeftModel 或裸模型里的解码层列表喵。

    输入: `PeftModel` 或 `Qwen3_5TextModel`
    输出: `ModuleList`，每个元素是一层解码层
    预期: 兼容 `PeftModel.base_model.model.layers` 与 `model.layers` 两种结构
    """
    inner = getattr(model, "base_model", model)
    inner = getattr(inner, "model", inner)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError("模型里找不到 .layers，结构与本项目预期不符喵")
    return layers


def collect_tap_hidden(layers, taps: tuple[int, ...], run) -> list[torch.Tensor]:
    """跑一次前向，只保留指定层的输出喵。

    输入: 解码层列表、抽头下标、触发前向并返回模型输出的零参函数
    输出: 与 `taps` 同序的张量列表
    预期: 与 `output_hidden_states=True` 下的 `hidden_states[t]` 是同一批张量

    Notes:
        实测这个模型的 `hidden_states[t]` 是**第 t 层的输入**，也就是第 `t - 1` 层的输出，
        所以正数抽头挂在该层输出的 hook 上；
        `hidden_states[-1]` 是过完 final norm 的 `last_hidden_state`，不是最后一层的裸输出，
        那一个必须从 `run()` 的返回值里取，挂 hook 会拿到错的东西喵。
        验证脚本里两者最大绝对差为 0，而用最后一层裸输出会差 3.73喵。
    """
    count = len(layers)
    wanted: dict[int, list[int]] = {}
    for position, tap in enumerate(taps):
        if tap < 0:
            continue
        if not 1 <= tap <= count - 1:
            raise IndexError(
                f"抽头 {tap} 无效：正数只能是 1 到 {count - 1}，"
                f"{count} 等于层数时对应 final norm，请用 -1 表示喵"
            )
        wanted.setdefault(tap - 1, []).append(position)

    captured: list[torch.Tensor | None] = [None] * len(taps)

    def make_hook(index: int):
        """给第 index 层造一个把输出写进 captured 的钩子喵。"""

        def hook(module, args, output):
            for position in wanted[index]:
                captured[position] = output

        return hook

    handles = [layers[index].register_forward_hook(make_hook(index)) for index in wanted]
    try:
        out = run()
    finally:
        for handle in handles:
            handle.remove()
    for position, tap in enumerate(taps):
        if tap < 0:
            captured[position] = out.last_hidden_state
    if any(item is None for item in captured):
        raise RuntimeError("有抽头没抓到输出，前向可能没走到对应层喵")
    return [item for item in captured if item is not None]


def build_lora(model, config: FinetuneConfig):
    """给已冻结的主干挂 LoRA，返回 PeftModel 与可训练参数量喵。

    输入: 已 `requires_grad_(False)` 的 Qwen3_5TextModel、配置
    输出: `(peft_model, trainable, total)`
    预期: 只有 LoRA 参数可训练，主干保持冻结喵
    """
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    return peft_model, trainable, total


class IdCache:
    """可选的预分词缓存喵。

    输入: 全量 token id（int32 一维数组）+ 每条 prompt 的偏移
    输出: `get(i)` 返回第 i 条 prompt 的 token id 视图
    预期: 训练时不再调用分词器；缓存文件小（1443 万 token 约 58 MB）

    Notes:
        分词与主干无关，训练三轮只需算一次喵。
    """

    def __init__(self, ids: np.ndarray, offsets: np.ndarray) -> None:
        self._ids = ids
        self._offsets = offsets

    @classmethod
    def load(cls, path: Path) -> "IdCache":
        """从 npz 读缓存喵。

        输入: 缓存路径
        输出: `IdCache`
        预期: 文件里必须有 `ids` 与 `offsets` 两个数组
        """
        data = np.load(str(path))
        return cls(data["ids"], data["offsets"])

    @classmethod
    def build(cls, tokenizer, texts: list[str], path: Path) -> "IdCache":
        """把全量 prompt 分词并落盘喵。

        输入: 分词器、全部文本、缓存路径
        输出: `IdCache`
        预期: 写 npz（`ids` int32、`offsets` int64），返回可用的缓存
        """
        lengths = [len(tokenizer.encode(t)) for t in texts]
        offsets = np.zeros(len(texts) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        ids = np.empty(int(offsets[-1]), dtype=np.int32)
        for i, text in enumerate(texts):
            ids[offsets[i]:offsets[i + 1]] = tokenizer.encode(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), ids=ids, offsets=offsets)
        return cls(ids, offsets)

    def get(self, index: int) -> np.ndarray:
        """取第 index 条 prompt 的 token id喵。"""
        return self._ids[self._offsets[index]:self._offsets[index + 1]]


def make_batch(
    tokenizer,
    texts: list[str],
    teachers: list[np.ndarray],
    id_seqs: list[np.ndarray] | None = None,
    pad_to: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """把若干条 prompt 打成右 padding 的一个批喵。

    输入: 分词器、文本列表、对应的教师特征、可选的预分词 id 列表、
        固定档位长度（0 表示用批内最长）
    输出: `(input_ids[B,T], attention_mask[B,T], teacher 列表)`
    预期: 两边都切掉 token 0（attention sink），保持逐行对齐喵

    Notes:
        raw 里保留了 token 0，训练端由读取层跳过；这里同样剔除，
        否则 sink 那个范数上万的行会把 CKA 与余弦都带偏喵。
        `id_seqs` 给出时直接用它，不再调用分词器喵。
        `pad_to` 大于批内最长时把 T 固定到档位：档位切批要求同一档
        的批形状恒为 `B_tier x T_tier`，张量尺寸种类才降得下来喵。
    """
    if id_seqs is None:
        # 逐条调 encode 时固定开销占主导，批量接口实测快 4.17 倍喵
        all_ids: Sequence[Sequence[int]] = tokenizer.encode_batch(texts)
    else:
        all_ids = id_seqs
    seqs: list[torch.Tensor] = []
    kept_teachers: list[torch.Tensor] = []
    for ids, teacher in zip(all_ids, teachers):
        if len(ids) != teacher.shape[0]:
            continue
        seqs.append(torch.tensor(ids[1:], dtype=torch.long))
        # 保持 fp16 上卡：CPU 上转 fp32 要多花一趟约 67 MB 的转换，
        # 而且 PCIe 传输量翻倍；fp32 提升留到 GPU 上做，几乎不花时间喵
        kept_teachers.append(torch.from_numpy(teacher[1:]))
    if not seqs:
        raise RuntimeError("没有一条样本的 token 数与特征对齐喵")

    max_len = max(int(s.shape[0]) for s in seqs)
    if pad_to > max_len:
        max_len = pad_to
    input_ids = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = 1
    return input_ids, mask, kept_teachers


def build_packed_batches(
    indices: list[int],
    lengths: list[int],
    budget: int,
    rng: np.random.Generator,
    max_prompts: int = 256,
) -> list[list[int]]:
    """按真实 token 数切批，打包不需要补齐喵。

    输入: 候选下标、每条 token 数、单批真实 token 上限、随机数发生器、单批条数上限
    输出: 批次列表，每批 `sum(lengths) <= budget`
    预期: 不再因为批内最长长度而多留 padding，预算按真实 token 填满喵

    Notes:
        补齐版按 `条数 x 批内最长` 控预算，短句桶里 20 token 的样本会被 pad 到 31，
        实测一个 epoch 白算 27 到 29 个百分点；打包版把这块省下来喵。
    """
    order = list(indices)
    rng.shuffle(order)
    batches: list[list[int]] = []
    batch: list[int] = []
    used = 0
    for index in order:
        count = int(lengths[index])
        if batch and (used + count > budget or len(batch) + 1 > max_prompts):
            batches.append(batch)
            batch, used = [], 0
        batch.append(index)
        used += count
    if batch:
        batches.append(batch)
    return batches


def make_packed_batch(
    tokenizer,
    texts: list[str],
    teachers: list[np.ndarray],
    id_seqs: list[np.ndarray] | None = None,
    pad_to: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """把若干条 prompt 首尾相接成一个批，可选把总长补齐到固定值喵。

    输入: 分词器、文本列表、对应教师特征、可选的预分词 id 列表、
        固定总长 `pad_to`（0 表示不补）
    输出: `(input_ids[1,total], cu_seqlens[N+1 或 N+2], teacher 列表)`
    预期: 两边都切掉 token 0，拼接顺序与输入一致；`pad_to` 大于真实长度时
        末尾追加一个只含 padding 的假段，返回的 `teacher` 不含这一段喵

    Notes:
        补齐是为了让每一步的激活张量形状完全一致。变长打包下每步总长与段数都在变，
        显存分配器只能反复找新块，保留量会一点点抬高，抬到物理上限就溢出到共享显存，
        表现为训练前期很快、某一步之后突然变慢喵。
        假段放在最末尾，训练侧按真实 token 数切掉前缀即可，不参与任何损失喵。
    """
    all_ids = tokenizer.encode_batch(texts) if id_seqs is None else id_seqs
    flat: list[int] = []
    bounds: list[int] = [0]
    kept: list[torch.Tensor] = []
    for ids, teacher in zip(all_ids, teachers):
        if len(ids) != teacher.shape[0]:
            continue
        body = [int(token) for token in ids[1:]]
        if not body:
            continue
        flat.extend(body)
        bounds.append(bounds[-1] + len(body))
        kept.append(torch.from_numpy(teacher[1:]))
    if not kept:
        raise RuntimeError("没有一条样本的 token 数与特征对齐喵")
    if pad_to > len(flat):
        flat.extend([PAD_ID] * (pad_to - len(flat)))
        bounds.append(len(flat))
    input_ids = torch.tensor(flat, dtype=torch.long).unsqueeze(0)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    return input_ids, cu_seqlens, kept


def distill_loss(
    hidden: torch.Tensor,
    teacher_flat: torch.Tensor,
    config: FinetuneConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """算蒸馏损失喵。

    输入: 学生 6 抽头拼接后的 `[N,d_in]`、教师 `[N,d_out]`、配置
    输出: `(loss, 分项字典)`，含 `loss` / `cka` / `collapse`
    预期: `loss = cka_weight·(1−CKA) + collapse_weight·塌缩正则`
    """
    loss_cka = cka_loss(
        hidden,
        teacher_flat,
        dtype=torch.bfloat16 if config.cka_bf16 else torch.float32,
    )
    loss_col = pairwise_cosine_penalty(hidden)

    total = config.cka_weight * loss_cka + config.collapse_weight * loss_col
    return total, {
        "loss": float(total.detach()),
        "cka": float(loss_cka.detach()),
        "collapse": float(loss_col.detach()),
    }


class PromptFeatureReader:
    """按 prompt 下标从 teacher.raw 里随机读特征，不把整份文件读进内存喵。

    输入: 特征目录、每条的 token 数、每条的起始行、教师维度
    输出: `read(indices)` 返回与下标顺序一一对应的 `[len, d_out]` float16 数组
    预期: 每次只 seek + read 需要的字节；一次训练批约 30~40 条，寻道开销可忽略喵

    Notes:
        把全部教师特征读成 list 放进内存，2.4M token 就是 19.6 GB，
        20.6M token 是 169 GB，等于必然爆内存喵。
    """

    def __init__(self, features: Path, lengths, starts, d_out: int) -> None:
        self._fp = (features / "teacher.raw").open("rb")
        self.lengths = lengths
        self._starts = starts
        self._d_out = d_out
        self._row_bytes = d_out * 2

    def read(self, indices) -> list[np.ndarray]:
        """读一批 prompt 的教师特征，按磁盘偏移升序发读请求喵。

        输入: prompt 下标序列
        输出: 与输入顺序一致的 float16 数组列表
        预期: 顺序随机但磁盘读按偏移单调，避免来回长距离寻道喵
        """
        order = sorted(range(len(indices)), key=lambda k: int(self._starts[indices[k]]))
        out: list[np.ndarray | None] = [None] * len(indices)
        for k in order:
            idx = int(indices[k])
            n = int(self.lengths[idx])
            self._fp.seek(int(self._starts[idx]) * self._row_bytes)
            buf = self._fp.read(n * self._row_bytes)
            if len(buf) != n * self._row_bytes:
                raise IOError(f"teacher.raw 读不足：prompt {idx} 期望 {n} 行喵")
            out[k] = np.frombuffer(buf, dtype=np.float16).reshape(n, self._d_out)
        return out  # type: ignore[return-value]

    def close(self) -> None:
        self._fp.close()
        return None


def build_length_batches(
    indices: list[int],
    lengths: list[int],
    budget: int,
    rng: np.random.Generator,
    max_prompts: int = 256,
    max_aspect: float = 3.0,
) -> list[list[int]]:
    """把 prompt 下标切成若干批，按**padding 后**的 token 数控制预算喵。

    输入: 候选下标、每条 token 数、单批「条数 x 批内最长」上限、随机数发生器、
        单批条数上限、批内最长/最短长度比上限
    输出: 批次列表，每批 `len(batch) * max(lengths in batch) <= budget`，
        且批内 `最长 / 最短 <= max_aspect`
    预期: 显存占用可预测；同批长度接近，padding 浪费小喵

    Notes:
        显存吃的是 `B x T_max`，不是真实 token 总和喵。
        按真实 token 数分批，短句桶里 200 条各 20 token 会被 pad 到 127，
        实际激活 25,400 token（超预算 6 倍），第一次前向就 CUDA OOM喵。
        只按长度桶分组不够：桶宽 32 时，1 token 和 32 token 仍会同批，
        补齐到 32 之后 1 token 的那条白烧 31 份算力。所以这里再按
        最长/最短比卡一次，取值与编码侧 `make_batches` 保持一致喵。
    """
    order = sorted(indices, key=lambda i: (int(lengths[i]), float(rng.random())))

    batches: list[list[int]] = []
    batch: list[int] = []
    longest = 0
    shortest = 0
    for idx in order:
        n = int(lengths[idx])
        new_longest = max(longest, n)
        new_shortest = min(shortest, n) if batch else n
        too_big = bool(batch) and (
            (len(batch) + 1) * new_longest > budget
            or len(batch) + 1 > max_prompts
            or new_longest > new_shortest * max_aspect
        )
        if too_big:
            batches.append(batch)
            batch, longest, shortest = [], 0, 0
            new_longest = n
            new_shortest = n
        batch.append(idx)
        longest = new_longest
        shortest = new_shortest
    if batch:
        batches.append(batch)

    # 批内已经按长度排好，但批与批之间要打乱，否则长度单调下降
    order_of_batches = rng.permutation(len(batches))
    return [batches[i] for i in order_of_batches]


@torch.no_grad()
def default_pad_tiers(batch_tokens: int) -> list[int]:
    """按 token 预算生成几何递增的默认档位喵。

    输入: 单步 token 预算
    输出: 档位长度列表（升序），相邻档位比值约 1.5，短段密、长段疏
    预期: 每档「条数 x 档长」恰为预算的倍数关系之内，单条最长样本
          独占一批时档长也盖得住喵

    Notes:
        几何递增让量化浪费有界：一条样本向上取整到最近档位，
        最多浪费约 1/1.5 ≈ 33% 的位置喵。
    """
    tiers = [16]
    while tiers[-1] < batch_tokens:
        nxt = min(batch_tokens, max(tiers[-1] + 1, int(tiers[-1] * 1.5)))
        tiers.append(nxt)
    return tiers


def build_tier_batches(
    indices: list[int],
    lengths: list[int],
    budget: int,
    rng: np.random.Generator,
    tiers: list[int],
    max_prompts: int = 256,
) -> list[list[int]]:
    """把 prompt 下标按**固定档位**切成批，批形状只有少数几种喵。

    输入: 候选下标、每条 token 数、单批 token 预算、随机数发生器、
        档位长度列表（升序）、单批条数上限
    输出: 批次列表；同一档位的批，形状恒为 `B_tier x T_tier`，
        `B_tier = max(1, budget // T_tier)`
    预期: 全训练的张量尺寸种类 <= 档位数，PyTorch 缓存池不再随步数增长喵

    Notes:
        病根是「批形状种类数 x 反向大池子」：变长批 500 步能造出几百种
        尺寸，resv 一路顶穿显存。这里把每条样本长度向上取整到档位，
        T 与 B 都由档位定死，形状种类骤减喵。
        条数还受 max_prompts 卡顶：短档位一条 16 token 的样本占不到
        预算的 1%，若不卡顶一批会塞进几百条短句喵。
    """
    tier_B = [max(1, budget // t) for t in tiers]
    slots: list[list[int]] = [[] for _ in tiers]
    # 每个档位记录「当前正在填的槽下标」：溢出后新槽继续接收样本，
    # 否则判断条件永远量第一个槽、第一批满了之后全部一条一批喵
    cur = list(range(len(tiers)))
    # 每个当前槽内已见的最短长度：同档内长短混装会把补齐预算提前耗尽，
    # 批碎成两三条喵；按教师侧 make_batches 的口径卡最长/最短比喵
    slot_min: list[int] = [0] * len(tiers)
    for idx in sorted(indices, key=lambda i: int(lengths[i])):
        n = int(lengths[idx])
        # 超过最大档位的样本独占一批，B=1，形状单独一种（数据里极少）
        if n > tiers[-1]:
            slots.append([idx])
            tier_B.append(1)
            continue
        ti = next(i for i, t in enumerate(tiers) if t >= n)
        k = tier_B[ti]
        s = slots[cur[ti]]
        # 长度比：槽里已有最短样本的 max_aspect 倍装不下这条，提前关批喵
        aspect_full = s and slot_min[ti] and n > slot_min[ti] * 3.0
        if s and (aspect_full
                  or (len(s) + 1) * tiers[ti] > budget
                  or len(s) + 1 > min(max_prompts, k)):
            slots.append([idx])
            tier_B.append(k)
            cur[ti] = len(slots) - 1
            slot_min[ti] = n
        elif not s:
            slots[cur[ti]].append(idx)
            slot_min[ti] = n
        else:
            slots[cur[ti]].append(idx)

    # 档位之间打乱，避免相邻步的形状与长度单调；空档位剔除喵
    order = rng.permutation(len(slots))
    return [slots[i] for i in order if slots[i]]


def _cka_fp32(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-8) -> float:
    """算一批 token 的线性 CKA喵。

    输入: 学生 `[N,dS]`、教师 `[N,dT]`，未去均值
    输出: float，0~1
    预期: 与 `cka_loss` 同一恒等式，不建图喵
    """
    s = student.reshape(-1, student.shape[-1]).float()
    t = teacher.reshape(-1, teacher.shape[-1]).float()
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(s.t() @ t)
    den_s = torch.linalg.matrix_norm(s.t() @ s) + eps
    den_t = torch.linalg.matrix_norm(t.t() @ t) + eps
    ratio = cross / torch.sqrt(den_s * den_t)
    return float(ratio * ratio)


def evaluate_cka(
    model,
    tokenizer,
    reader: PromptFeatureReader,
    texts: list[str],
    indices: list[int],
    taps: tuple[int, ...],
    device: torch.device,
    budget: int,
    id_cache: "IdCache | None" = None,
) -> float:
    """在验证集上算学生与教师隐状态的线性 CKA喵。

    输入: 主干、分词器、特征读取器、文本表、验证下标、抽头层、设备、单批 token 上限
    输出: 按 token 数加权的 CKA 均值（0~1，越大越好）
    预期: 不需要线性头，两边维度不同也能比
    """
    if not indices:
        return float("nan")
    model.eval()
    sum_cka = 0.0
    n_tok = 0
    with torch.no_grad():
        batches = build_length_batches(indices, reader.lengths, budget, np.random.default_rng(0))
        for batch in batches:
            teachers = reader.read(batch)
            try:
                id_seqs = [id_cache.get(i) for i in batch] if id_cache is not None else None
                input_ids, mask, kept = make_batch(
                    tokenizer, [texts[i] for i in batch], teachers, id_seqs
                )
            except RuntimeError:
                continue
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            def run_forward():
                """跑一次前向，抽头交给 hook 收集喵。"""
                return model(input_ids=input_ids, attention_mask=mask, use_cache=False)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                taps_out = collect_tap_hidden(decoder_layers(model), taps, run_forward)
                hidden = torch.cat(taps_out, dim=-1).float()
            hidden = hidden[mask.bool()]
            teacher_flat = torch.cat(
                [t[: int(n)] for t, n in zip(kept, mask.sum(1).tolist())], dim=0
            ).to(device)
            sum_cka += _cka_fp32(hidden, teacher_flat) * int(hidden.shape[0])
            n_tok += int(hidden.shape[0])
    model.train()
    return sum_cka / max(n_tok, 1)


class _BatchPrefetcher:
    """在后台线程准备训练批，让读盘与分词和 GPU 计算重叠喵。

    输入: 特征读取器、分词器、文本表、可选的预分词缓存、批次列表
    输出: 迭代 `(batch, prepared)`；`prepared` 为 `None` 表示该批 token 数与特征对不上
    预期: 主线程拿到 CPU 张量后立刻上卡，后台同时准备下一批；
          后台只做 CPU 侧工作、不碰 CUDA，避免多线程上下文问题喵
    """

    def __init__(
        self,
        reader: "PromptFeatureReader",
        tokenizer,
        texts: list[str],
        id_cache: "IdCache | None",
        batches,
        depth: int = 2,
        packed: bool = False,
    ) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._packed = packed
        self._thread = threading.Thread(
            target=self._produce,
            args=(reader, tokenizer, texts, id_cache, batches),
            daemon=True,
        )

    def start(self) -> "_BatchPrefetcher":
        """启动后台线程并返回自身喵。"""
        self._thread.start()
        return self

    def _produce(self, reader, tokenizer, texts, id_cache, batches) -> None:
        """后台循环：读一批、分一批、塞进队列喵。

        输入: 读取器、分词器、文本表、预分词缓存、批次列表（`(批, pad_to)` 二元组）
        输出: 无（写队列）
        预期: 队列满时阻塞等主线程取走；结束时塞一个 `None` 当哨兵喵
        """
        for batch, pad_to in batches:
            if self._stop.is_set():
                break
            teachers = reader.read(batch)
            id_seqs = [id_cache.get(i) for i in batch] if id_cache is not None else None
            try:
                if self._packed:
                    prepared = make_packed_batch(
                        tokenizer, [texts[i] for i in batch], teachers, id_seqs,
                        pad_to=pad_to,
                    )
                else:
                    prepared = make_batch(
                        tokenizer, [texts[i] for i in batch], teachers, id_seqs,
                        pad_to=pad_to,
                    )
            except RuntimeError:
                prepared = None
            except Exception:
                # 后台线程异常若直接逃逸，主线程会永远等不到哨兵而假死；
                # 这里统一降级成「该批不可用」，训练循环按 skipped 处理喵
                self._queue.put((batch, None))
                continue
            self._queue.put((batch, prepared))
        try:
            self._queue.put((None, None), timeout=1.0)
        except queue.Full:
            pass

    def __iter__(self):
        """按批产出 `(batch, prepared)`，遇到哨兵结束喵。"""
        while True:
            batch, prepared = self._queue.get()
            if batch is None:
                return
            yield batch, prepared

    def stop(self) -> None:
        """通知后台线程停下，并等它真正退出喵。

        输入: 无
        输出: 无
        预期: 返回时后台线程已结束，调用方随后关闭特征文件才是安全的喵

        Notes:
            队列满时后台线程会阻塞在 `put` 上，先抽干队列让它走到下一次停止判断喵。
        """
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """找输出目录里步数最大的完整断点喵。

    输入: 输出目录
    输出: 最新 `checkpoint_step_*.pt` 的路径；没有则 None
    预期: 轮转删旧后剩下的就是最新那份，`--resume` 不带值时用它喵
    """
    ckpts = sorted(out_dir.glob("checkpoint_step_*.pt"))
    return ckpts[-1] if ckpts else None


def train_finetune(
    model,
    tokenizer,
    reader: PromptFeatureReader,
    texts: list[str],
    train_idx: list[int],
    val_idx: list[int],
    config: FinetuneConfig,
    out_dir: Path,
    device: str | torch.device = "cuda",
    id_cache: "IdCache | None" = None,
    resume: Path | None = None,
) -> FinetuneResult:
    """跑流式微调循环喵。

    输入: 主干、分词器、特征读取器、全部文本、训练/验证下标、配置、输出目录、
        断点路径（None 表示从头训）
    输出: `FinetuneResult`；LoRA 写到 `out_dir/lora`
    预期: 每步现读一小批教师特征，显存与内存都不随语料规模增长喵

    Notes:
        只训 LoRA，优化器只有一组参数；线性头留给训练之后单独解喵。
        `resume` 给出时恢复 step/优化器/调度器，LoRA 权重从
        `lora_step_XXXXXX/` 同名目录加载；断点里的步数之前的批被跳过，
        批次顺序由固定种子保证与首跑一致喵。
    """
    device = torch.device(device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    peft_model, trainable, total = build_lora(model, config)
    print(f"LoRA 可训练 {trainable:,} / 总 {total:,} = {trainable / total:.3%}", flush=True)

    # 物理显存上限：reserved 是 PyTorch 自己申请的总量喵，越过它说明 WDDM
    # 正在把超出的部分塞进共享显存（无声掉速的根源）喵，触发即自杀喵
    vram_total = torch.cuda.get_device_properties(device).total_memory

    def _snapshot_dump(path: Path) -> None:
        """把每个活跃分配块的调用栈写进 json，显存账单喵。"""
        snap = torch.cuda.memory_snapshot()
        out = []
        for seg in snap:
            for b in seg.get("blocks", []):
                if b["state"] != "active_allocated":
                    continue
                frames = [f.get("filename", "") + ":" + str(f.get("line", "")) + " " + f.get("name", "")
                          for f in (b.get("frames") or [])[:4]]
                out.append({"size": b["size"], "frames": frames})
        out.sort(key=lambda item: -item["size"])
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"显存快照已写 {path.name}（{len(out)} 块）喵", flush=True)

    # 不开历史记录时快照没有调用栈（frames 全空喵）；只在越界 dump 前临时开启喵
    # 训练开始即记录喵：需要覆盖从第一步起的分配喵
    torch.cuda.memory._record_memory_history(max_entries=65536, stack_depth=4)

    def _check_vram(step: int) -> None:
        resv = torch.cuda.memory_reserved(device)
        if resv <= vram_total:
            return
        print(f"第 {step} 步 resv {resv / 2**30:.2f} GiB 超过物理显存 "
              f"{vram_total / 2**30:.2f} GiB，已溢出到共享显存，立即停止喵", flush=True)
        report("over-vram")
        _snapshot_dump(out_dir / "mem_snapshot.json")
        raise SystemExit(1)

    def report(tag: str) -> None:
        """打印当前显存 top10 活跃块及其分配调用栈喵。"""
        snap = torch.cuda.memory_snapshot()
        blocks = []
        for seg in snap:
            for b in seg.get("blocks", []):
                if b["state"] == "active_allocated":
                    frames = [f.get("name", "") for f in (b.get("frames") or [])[:2]]
                    blocks.append((b["size"], frames))
        blocks.sort(key=lambda item: -item[0])
        top = " | ".join(f"{s / 2**20:.0f}M@{frames[0] if frames else '?'}" for s, frames in blocks[:10])
        print(f"[{tag}] alloc {torch.cuda.memory_allocated(device) / 2**30:.2f} "
              f"resv {torch.cuda.memory_reserved(device) / 2**30:.2f} top[{top}]", flush=True)

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": config.lr_lora, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.steps, eta_min=config.lr_lora * config.min_lr_ratio
    )

    start_step = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        start_step = int(payload["step"])
        lora_dir = resume.parent / f"lora_step_{start_step:06d}"
        # peft 存的键带 base_model.model. 前缀，须用 PeftModel.from_pretrained 装喵
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(
            model, str(lora_dir), is_trainable=True,
            adapter_name="default",
        )
        peft_model.to(device)
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        print(f"从 {resume.name} 恢复到第 {start_step} 步（LoRA 来自 {lora_dir.name}）", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = FinetuneResult(steps=0)
    peft_model.train()
    started = time.time()
    last_log_t, last_log_step = started, start_step
    best = -1.0
    step = start_step
    skipped = 0

    while step < config.steps:
        if config.packed:
            batches = build_packed_batches(train_idx, reader.lengths, config.batch_tokens, rng)
            work = [(b, config.batch_tokens) for b in batches]
            prefetcher = _BatchPrefetcher(
                reader, tokenizer, texts, id_cache, work, packed=True,
            ).start()
        else:
            batches = build_tier_batches(
                train_idx, reader.lengths, config.batch_tokens, rng, config.pad_tiers
            )
            # 每批带着自己的档位走：同档所有批 pad 到同一 T，形状恒为
            # B_tier x T_tier，分配器看到的尺寸种类就是档位数喵
            tier_map = {t: max(1, config.batch_tokens // t) for t in config.pad_tiers}
            work: list[tuple[list[int], int]] = []
            for batch in batches:
                longest = max(reader.lengths[i] for i in batch)
                tier = next((x for x in config.pad_tiers if x >= longest), longest)
                if longest > config.pad_tiers[-1]:
                    tier = longest  # 超档位独占批，按批内最长补
                elif len(batch) < min(256, tier_map[tier]):
                    # 未填满的档尾批也补到档位，形状种类不增喵
                    pass
                work.append((batch, tier))
            prefetcher = _BatchPrefetcher(
                reader, tokenizer, texts, id_cache, work,
            ).start()
        for _batch, prepared in prefetcher:
            if step >= config.steps:
                prefetcher.stop()
                break
            if step < start_step:
                # resume：首轮里已完成步对应的批直接跳过，不读特征不算损失喵
                step += 1
                continue
            if prepared is None:
                skipped += 1
                continue
            if config.packed:
                input_ids, cu_seqlens, kept = prepared
                input_ids = input_ids.to(device)
                cu_seqlens = cu_seqlens.to(device)
                packed_kwargs = packed_forward_kwargs(cu_seqlens)

                def run_forward():
                    """打包前向：整批是一条序列喵。"""
                    return peft_model(
                        input_ids=input_ids, use_cache=False, **packed_kwargs
                    )

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    taps_out = collect_tap_hidden(
                        decoder_layers(peft_model), config.taps, run_forward
                    )
                    hidden_full = torch.cat(taps_out, dim=-1).float()
                # 末尾那段是补齐用的假段，按真实 token 数切掉，不进损失喵
                real_tokens = sum(int(item.shape[0]) for item in kept)
                hidden = hidden_full.reshape(-1, hidden_full.shape[-1])[:real_tokens]
                teacher_flat = torch.cat([item.to(device) for item in kept], dim=0)
            else:
                input_ids, mask, kept = prepared
                input_ids = input_ids.to(device)
                mask = mask.to(device)

                def run_forward():
                    """补齐前向：抽头交给 hook 收集喵。"""
                    return peft_model(
                        input_ids=input_ids, attention_mask=mask, use_cache=False
                    )

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    taps_out = collect_tap_hidden(
                        decoder_layers(peft_model), config.taps, run_forward
                    )
                    hidden_full = torch.cat(taps_out, dim=-1).float()
                hidden = hidden_full[mask.bool()]
                teacher_flat = torch.cat(
                    [t[: int(n)] for t, n in zip(kept, mask.sum(1).tolist())], dim=0
                ).to(device)

            # 损失必须在 autocast 之外算：CKA 走 float32，余弦要稳定喵
            loss, parts = distill_loss(hidden, teacher_flat, config)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            opt.step()
            sched.step()
            step += 1
            del taps_out, hidden_full, hidden, teacher_flat, loss

            if step % config.log_every == 0 or step == 1:
                now = time.time()
                record = {"step": step, **parts, "sec": round(now - started, 1)}
                result.history.append(record)
                sec_per_step = (now - last_log_t) / (step - last_log_step)
                print(f"step {step:>5} loss {parts['loss']:.6f} cka {parts['cka']:.6f} "
                      f"col {parts['collapse']:.4f} "
                      f"{sec_per_step:.2f}s/step "
                      f"alloc {torch.cuda.memory_allocated() / 2**30:.2f} "
                      f"resv {torch.cuda.memory_reserved() / 2**30:.2f}", flush=True)
                _check_vram(step)
                if step % (config.log_every * 10) == 0 or step == config.log_every:
                    report(f"step {step}")
                last_log_t, last_log_step = now, step

            if config.eval_every and step % config.eval_every == 0:
                cka = evaluate_cka(peft_model, tokenizer, reader, texts, val_idx,
                                   config.taps, device, config.batch_tokens, id_cache)
                print(f"  验证 step {step} 线性 CKA {cka:.5f}", flush=True)
                result.history.append({"step": step, "val_cka": cka})
                if cka > best:
                    best = cka
                    peft_model.save_pretrained(str(out_dir / "lora_best"))
                    print(f"  最优更新 -> {cka:.5f}", flush=True)

            if config.save_every and step % config.save_every == 0:
                save_checkpoint(peft_model, opt, sched, step, out_dir, config.keep_last)

    result.steps = step
    peft_model.save_pretrained(str(out_dir / "lora"))
    if skipped:
        print(f"跳过 {skipped} 批（分词长度与特征行数对不上）", flush=True)
    print(f"完成 {step} 步，用时 {time.time() - started:.1f}s -> {out_dir}", flush=True)
    return result


@torch.no_grad()
def self_check(
    model,
    tokenizer,
    texts: list[str],
    teachers: list[np.ndarray],
    config: FinetuneConfig,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """训练前自检喵。

    输入: 未挂 LoRA 的主干、分词器、样本、教师特征、配置
    输出: 含 `lora_max_diff`、`cka`、`cka_numpy` 的字典，`cka` 与 `cka_numpy`
        都是损失口径 `1 − √CKA`
    预期: `lora_max_diff` 为 0.0（LoRA 零初始化不改变输出），
          且训练路径的 CKA 与 numpy 独立实现相差小于 1e-4
    """
    device = torch.device(device)
    input_ids, mask, kept = make_batch(tokenizer, texts, teachers)
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    kept = [t.to(device) for t in kept]

    def run_base():
        """不挂 LoRA 的前向喵。"""
        return model(input_ids=input_ids, attention_mask=mask, use_cache=False)

    base_taps = collect_tap_hidden(decoder_layers(model), config.taps, run_base)
    base_h = torch.cat(base_taps, dim=-1).float()
    del base_taps

    peft_model, trainable, total = build_lora(model, config)
    peft_model.eval()

    def run_lora():
        """挂了 LoRA 的前向喵。"""
        return peft_model(input_ids=input_ids, attention_mask=mask, use_cache=False)

    lora_taps = collect_tap_hidden(decoder_layers(peft_model), config.taps, run_lora)
    lora_h = torch.cat(lora_taps, dim=-1).float()
    del lora_taps
    max_diff = float((lora_h - base_h).abs().max())

    hidden = base_h[mask.bool()]
    teacher_flat = torch.cat(
        [t[: int(n)] for t, n in zip(kept, mask.sum(1).tolist())], dim=0
    )

    xs = hidden.detach().cpu().numpy().astype(np.float64)
    ys = teacher_flat.detach().cpu().numpy().astype(np.float64)
    xs = xs - xs.mean(axis=0, keepdims=True)
    ys = ys - ys.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(ys.T @ xs, "fro")
    den = np.sqrt(np.linalg.norm(xs.T @ xs, "fro") * np.linalg.norm(ys.T @ ys, "fro"))
    # `distill_loss` 返回的是损失口径 `1 − √CKA`，这里必须换算到同一口径再比喵
    cka_numpy = float((cross / den) ** 2)
    cka_numpy_loss = 1.0 - float(np.sqrt(cka_numpy))

    _, parts = distill_loss(hidden, teacher_flat, config)
    parts["lora_max_diff"] = max_diff
    parts["trainable"] = float(trainable)
    parts["total"] = float(total)
    parts["cka_numpy"] = cka_numpy_loss
    return parts


def save_checkpoint(
    peft_model, opt, sched, step: int, out_dir: Path, keep_last: int = 0
) -> None:
    """存一份断点，含 LoRA、优化器与调度器状态喵。

    输入: PeftModel、优化器、调度器、当前步、输出目录、保留最近几份断点
    输出: 落盘 `checkpoint_step_XXXXXX.pt` 与 `lora_step_XXXXXX/`，
        `keep_last > 0` 时删掉更早的 `checkpoint_step_*.pt`（LoRA 目录保留）
    预期: 断电最多丢 `save_every` 步；磁盘占用被 `keep_last` 封顶喵

    Notes:
        完整断点（63 MB/份）只服务 `--resume`，旧的根本用不上；
        LoRA 目录（31 MB/份）是训练产物本身，全保留喵。
    """
    path = out_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
        },
        path,
    )
    peft_model.save_pretrained(str(out_dir / f"lora_step_{step:06d}"))
    if keep_last > 0:
        olds = sorted(out_dir.glob("checkpoint_step_*.pt"))
        for old in olds[:-keep_last] if len(olds) > keep_last else []:
            old.unlink(missing_ok=True)
    print(f"  checkpoint -> {path.name}", flush=True)


def merge_and_save(peft_model, out_dir: Path, student_dir: Path | None = None) -> Path:
    """把 LoRA 合并进主干并落盘成可直接被编码器读取的 safetensors喵。

    输入: 已训练的 PeftModel、输出目录、原学生目录（用来复制 config.json）
    输出: `out_dir/merged/`，含 `model.safetensors` 与 `config.json`
    预期: 键名带 `model.language_model.` 前缀，`StudentEncoder` 能直接加载喵

    Notes:
        `save_pretrained` 写出的是裸键（`embed_tokens.weight`），而编码器
        只读带前缀的键；直接用它会导致权重全部被跳过、state 为空喵。
    """
    import shutil

    from safetensors.torch import save_file

    from moss_small_encoder.encode.common import MODEL_PREFIX

    merged = peft_model.merge_and_unload()
    merged_dir = out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    state = merged.state_dict()
    prefixed = {
        f"{MODEL_PREFIX}{key}": value.detach().to("cpu").contiguous()
        for key, value in state.items()
    }
    n_keys = len(prefixed)
    save_file(prefixed, str(merged_dir / "model.safetensors"))
    del state, prefixed

    if student_dir is not None:
        config_src = Path(student_dir) / "config.json"
        if config_src.is_file():
            shutil.copy2(config_src, merged_dir / "config.json")
    print(f"合并权重 -> {merged_dir}（{n_keys} 键已写）", flush=True)
    return merged_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="微调 0.8B 主干的训练循环喵")
    parser.add_argument("--features", type=Path, required=True, help="特征目录（含 meta.json）")
    parser.add_argument("--student", type=Path, default=Path("models/qwen3.5-0.8b"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer/tokenizer.json"))
    parser.add_argument("--prompts", type=Path, default=None,
                        help="prompts.jsonl；默认取 features 同级的 prompts.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=0,
                        help="过拟合实验：只取前 N 条训练样本；0 表示全部训练集")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=0,
                        help="总步数；0 表示按 epochs x 训练 token / batch_tokens 自动算")
    parser.add_argument("--batch-tokens", type=int, default=4096,
                        help="单步 token 预算（不含 padding）")
    parser.add_argument("--packed", action="store_true",
                        help="训练批也走变长打包、不补 padding；需要 flash-attn")
    parser.add_argument("--pad-tiers", default=None,
                        help="档位切批的固定长度档，逗号分隔（如 128,256,512,1024,2048）；"
                             "默认按 batch-tokens 自动生成几何档喵")
    parser.add_argument("--val-tokens", type=int, default=40000,
                        help="验证集最多用多少 token")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="每多少步算一次验证余弦；0 表示不评估")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--cka-fp32", action="store_false", dest="cka_bf16",
                        help="CKA 的三次 Gram 改回纯 fp32（默认走 bf16 输入 + fp32 累加）")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0,
                        help="每多少步存一次断点；0 表示只在最优点与结束时存")
    parser.add_argument("--keep-last", type=int, default=0,
                        help="断点轮转：只保留最近 N 份 checkpoint_step_*.pt；"
                             "0 表示全保留。LoRA 目录不受影响，全保留喵")
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="从断点恢复：给 checkpoint 路径，或不带值自动找最新喵")
    parser.add_argument("--merge", action="store_true",
                        help="把 LoRA 合并进主干并写到 --out/merged 喵")
    parser.add_argument("--lora", default="lora",
                        help="合并哪个子目录的 LoRA（默认 lora；验证最优点在 lora_best）")
    parser.add_argument("--check", action="store_true",
                        help="只做训练前自检：LoRA 零初始化 + 两条独立路径复算损失喵")
    parser.add_argument("--ids-cache", type=Path, default=None,
                        help="可选：预分词缓存 npz 路径；给出时缺文件则先建，之后训练不再分词")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from moss_small_encoder.encode.student import StudentEncoder
    from moss_small_encoder.tokenizer import MossTokenizer

    features = args.features
    meta = json.loads((features / "meta.json").read_text(encoding="utf-8"))
    ids = list(meta["prompt_ids"])
    lengths = [int(v) for v in meta["lengths"]]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    taps = tuple(int(t) for t in (meta.get("taps") or TAPS_DEFAULT))
    d_out = int(meta["d_out"])

    prompts_path = args.prompts or (features.parent / "prompts.jsonl")
    text_by_id: dict[str, str] = {}
    split_by_id: dict[str, str] = {}
    with prompts_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            text_by_id[row["id"]] = row["text"]
            split_by_id[row["id"]] = row.get("split", "train")

    texts = [text_by_id[pid] for pid in ids]
    splits = [split_by_id.get(pid, "train") for pid in ids]
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx: list[int] = []
    used = 0
    for i, s in enumerate(splits):
        if s != "val":
            continue
        if used + lengths[i] > args.val_tokens:
            break
        val_idx.append(i)
        used += lengths[i]
    if args.samples > 0:
        train_idx = train_idx[: args.samples]
        val_idx = val_idx[: max(8, args.samples // 4)]

    tiers: tuple[int, ...] | None = None
    if not args.packed:
        if args.pad_tiers:
            tiers = tuple(int(t) for t in args.pad_tiers.split(",") if t.strip())
            if any(t1 >= t2 for t1, t2 in zip(tiers, tiers[1:])):
                print("档位必须严格升序喵", flush=True)
                return 2
        else:
            tiers = tuple(default_pad_tiers(args.batch_tokens))
        print(f"档位 {list(tiers)}", flush=True)

    train_tokens = sum(lengths[i] for i in train_idx)
    # 步数按真实批数算：短句多的一轮批数远多于「token/预算」，不能拿理论值估喵
    if args.packed:
        plan = build_packed_batches(
            train_idx, lengths, args.batch_tokens, np.random.default_rng(0)
        )
    else:
        plan = build_tier_batches(
            train_idx, lengths, args.batch_tokens, np.random.default_rng(0),
            list(tiers),
        )
    steps = args.steps or max(1, len(plan) * max(1, args.epochs))
    print(f"训练 {len(train_idx)} 条 / {train_tokens:,} token；"
          f"验证 {len(val_idx)} 条 / {used:,} token；"
          f"每轮 {len(plan)} 批，共 {steps} 步 x {args.batch_tokens} token", flush=True)

    reader = PromptFeatureReader(features, lengths, starts, d_out)
    enc = StudentEncoder(
        args.student,
        device=args.device,
        dtype=torch.bfloat16,
        attn="flash_attention_2" if args.packed else "sdpa",
    )
    tokenizer = MossTokenizer.from_file(args.tokenizer)

    id_cache = None
    if args.ids_cache is not None:
        if args.ids_cache.is_file():
            id_cache = IdCache.load(args.ids_cache)
            print(f"预分词缓存命中 {args.ids_cache}", flush=True)
        else:
            id_cache = IdCache.build(tokenizer, texts, args.ids_cache)
            print(f"预分词缓存建好 {args.ids_cache}（{int(id_cache._offsets[-1]):,} token）", flush=True)

    config = FinetuneConfig(
        taps=taps,
        steps=steps,
        batch_tokens=args.batch_tokens,
        packed=args.packed,
        pad_tiers=tiers,
        eval_every=args.eval_every,
        lora_r=args.lora_r,
        lr_lora=args.lr,
        cka_bf16=args.cka_bf16,
        log_every=args.log_every,
        save_every=args.save_every,
        keep_last=args.keep_last,
    )
    if args.merge:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(enc.model, str(args.out / args.lora))
        merged = merge_and_save(peft_model, args.out, args.student)
        print(f"合并完成 -> {merged}", flush=True)
        reader.close()
        return 0
    if args.check:
        pick = (val_idx or train_idx)[:16]
        teachers = reader.read(pick)
        chk = self_check(
            enc.model, tokenizer, [texts[i] for i in pick], teachers,
            config, args.device
        )
        print(f"训练路径 cka {chk['cka']:.6f}   numpy 复算 {chk['cka_numpy']:.6f}", flush=True)
        print(f"LoRA 零初始化最大差 {chk['lora_max_diff']:.3e}   "
              f"可训练 {int(chk['trainable']):,} / {int(chk['total']):,}", flush=True)
        d_cka = abs(chk["cka"] - chk["cka_numpy"])
        ok = chk["lora_max_diff"] == 0.0 and d_cka < 1e-4
        print(f"自检{'通过' if ok else '失败'}  cka 差 {d_cka:.2e}", flush=True)
        reader.close()
        return 0 if ok else 1
    resume_path: Path | None = None
    if args.resume:
        if args.resume == "latest":
            resume_path = find_latest_checkpoint(args.out)
            if resume_path is None:
                print(f"{args.out} 下没有断点，从头训喵", flush=True)
        else:
            resume_path = Path(args.resume)
            assert resume_path.is_file(), f"断点不存在: {resume_path}"
    result = train_finetune(
        enc.model, tokenizer, reader, texts, train_idx, val_idx,
        config, args.out, args.device, id_cache, resume_path
    )
    reader.close()

    (args.out / "history.jsonl").write_text(
        "\n".join(json.dumps(rec) for rec in result.history) + "\n", encoding="utf-8"
    )
    print(f"首步 {result.first}", flush=True)
    print(f"末步 {result.last}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
