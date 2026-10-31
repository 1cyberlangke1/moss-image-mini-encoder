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
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .cka import cka_loss, pairwise_cosine_penalty

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
    lr_head: float = 1e-4
    steps: int = 200
    batch_tokens: int = 4096
    eval_every: int = 0
    cka_weight: float = 1.0
    cos_weight: float = 0.5
    collapse_weight: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 50
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


def make_batch(
    tokenizer,
    texts: list[str],
    teachers: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """把若干条 prompt 打成右 padding 的一个批喵。

    输入: 分词器、文本列表、对应的教师特征
    输出: `(input_ids[B,T], attention_mask[B,T], teacher 列表)`
    预期: 两边都切掉 token 0（attention sink），保持逐行对齐喵

    Notes:
        raw 里保留了 token 0，训练端由读取层跳过；这里同样剔除，
        否则 sink 那个范数上万的行会把 CKA 与余弦都带偏喵。
    """
    seqs: list[torch.Tensor] = []
    kept_teachers: list[torch.Tensor] = []
    for text, teacher in zip(texts, teachers):
        ids = tokenizer.encode(text)
        if len(ids) != teacher.shape[0]:
            continue
        seqs.append(torch.tensor(ids[1:], dtype=torch.long))
        kept_teachers.append(torch.from_numpy(teacher[1:].astype(np.float32)))
    if not seqs:
        raise RuntimeError("没有一条样本的 token 数与特征对齐喵")

    max_len = max(int(s.shape[0]) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = 1
    return input_ids, mask, kept_teachers


def distill_loss(
    hidden: torch.Tensor,
    teacher_flat: torch.Tensor,
    head: torch.Tensor,
    stats: dict[str, torch.Tensor],
    config: FinetuneConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """算蒸馏损失与三个分项喵。

    输入: 学生 6 抽头拼接后的 `[N,d_in]`、教师 `[N,d_out]`、线性头、标准化统计量
    输出: `(loss, 分项字典)`
    预期: 分项都降下去才算真的学到，只降一项通常是塌缩的假象喵
    """
    mean_in = stats["mean_in"]
    std_in = stats["std_in"]
    mean_out = stats["mean_out"]
    std_out = stats["std_out"]

    h_std = (hidden - mean_in) / std_in
    t_std = (teacher_flat - mean_out) / std_out

    loss_cka = cka_loss(h_std, t_std)
    pred = h_std @ head * std_out + mean_out
    loss_cos = (1.0 - F.cosine_similarity(pred, teacher_flat, dim=-1)).mean()
    loss_col = pairwise_cosine_penalty(h_std)

    total = (
        config.cka_weight * loss_cka
        + config.cos_weight * loss_cos
        + config.collapse_weight * loss_col
    )
    return total, {
        "loss": float(total.detach()),
        "cka": float(loss_cka.detach()),
        "cos_loss": float(loss_cos.detach()),
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
) -> list[list[int]]:
    """把 prompt 下标切成若干批，按**padding 后**的 token 数控制预算喵。

    输入: 候选下标、每条 token 数、单批「条数 x 批内最长」上限、随机数发生器、单批条数上限
    输出: 批次列表，每批 `len(batch) * max(lengths in batch) <= budget`
    预期: 显存占用可预测；同批长度接近，padding 浪费小喵

    Notes:
        显存吃的是 `B x T_max`，不是真实 token 总和喵。
        按真实 token 数分批，短句桶里 200 条各 20 token 会被 pad 到 127，
        实际激活 25,400 token（超预算 6 倍），第一次前向就 CUDA OOM喵。
    """
    keys = []
    for idx in indices:
        bucket = int(lengths[idx]) // 32
        keys.append((bucket, float(rng.random()), idx))
    keys.sort()

    batches: list[list[int]] = []
    batch: list[int] = []
    longest = 0
    for _, _, idx in keys:
        n = int(lengths[idx])
        new_longest = max(longest, n)
        too_big = batch and ((len(batch) + 1) * new_longest > budget
                             or len(batch) + 1 > max_prompts)
        if too_big:
            batches.append(batch)
            batch, longest = [], 0
            new_longest = n
        batch.append(idx)
        longest = new_longest
    if batch:
        batches.append(batch)
    return batches


def evaluate_cosine(
    model,
    tokenizer,
    reader: PromptFeatureReader,
    texts: list[str],
    indices: list[int],
    stats: dict[str, torch.Tensor],
    head: torch.Tensor,
    taps: tuple[int, ...],
    device: torch.device,
    budget: int,
) -> float:
    """在验证集上算投影后逐 token 余弦均值喵。

    输入: 主干、分词器、特征读取器、文本表、验证下标、标准化统计量、线性头、
          抽头层、设备、单批 token 上限
    输出: 余弦均值（-1~1，越大越好）
    预期: 与 train/run.py 的验证口径一致，都是原始空间逐 token 余弦喵
    """
    if not indices:
        return float("nan")
    model.eval()
    cos_sum = 0.0
    n_tok = 0
    with torch.no_grad():
        batches = build_length_batches(indices, reader.lengths, budget, np.random.default_rng(0))
        for batch in batches:
            teachers = reader.read(batch)
            try:
                input_ids, mask, kept = make_batch(tokenizer, [texts[i] for i in batch], teachers)
            except RuntimeError:
                continue
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=mask, use_cache=False,
                            output_hidden_states=True)
            hidden = torch.cat([out.hidden_states[t] for t in taps], dim=-1).float()
            hidden = hidden[mask.bool()]
            teacher_flat = torch.cat(
                [t[: int(n)] for t, n in zip(kept, mask.sum(1).tolist())], dim=0
            ).to(device)
            pred = (hidden - stats["mean_in"]) / stats["std_in"] @ head * stats["std_out"] + stats["mean_out"]
            cos = F.cosine_similarity(pred, teacher_flat, dim=-1)
            cos_sum += float(cos.sum())
            n_tok += int(cos.numel())
    model.train()
    return cos_sum / max(n_tok, 1)


def train_finetune(
    model,
    tokenizer,
    reader: PromptFeatureReader,
    texts: list[str],
    train_idx: list[int],
    val_idx: list[int],
    adapter_path: Path,
    config: FinetuneConfig,
    out_dir: Path,
    device: str | torch.device = "cuda",
) -> FinetuneResult:
    """跑流式微调循环喵。

    输入: 主干、分词器、特征读取器、全部文本、训练/验证下标、配置、输出目录
    输出: `FinetuneResult`；LoRA 写到 `out_dir/lora`
    预期: 每步现读一小批教师特征，显存与内存都不随语料规模增长喵

    Notes:
        只训 LoRA，优化器只有一组参数；线性头留给训练之后单独解喵。
    """
    device = torch.device(device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    peft_model, trainable, total = build_lora(model, config)
    print(f"LoRA 可训练 {trainable:,} / 总 {total:,} = {trainable / total:.3%}", flush=True)

    adapter = load_file(str(adapter_path))
    stats = {
        "mean_in": adapter["mean_in"].to(device, torch.float32),
        "std_in": adapter["std_in"].to(device, torch.float32),
        "mean_out": adapter["mean_out"].to(device, torch.float32),
        "std_out": adapter["std_out"].to(device, torch.float32),
    }
    head = torch.nn.Parameter(adapter["weight"].to(device, torch.float32))

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": params, "lr": config.lr_lora, "weight_decay": 0.0},
            {"params": [head], "lr": config.lr_head, "weight_decay": 0.0},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.steps, eta_min=config.lr_lora * config.min_lr_ratio
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = FinetuneResult(steps=0)
    peft_model.train()
    started = time.time()
    best = -1.0
    step = 0
    skipped = 0

    while step < config.steps:
        batches = build_length_batches(train_idx, reader.lengths, config.batch_tokens, rng)
        for batch in batches:
            if step >= config.steps:
                break
            teachers = reader.read(batch)
            try:
                input_ids, mask, kept = make_batch(
                    tokenizer, [texts[i] for i in batch], teachers
                )
            except RuntimeError:
                skipped += 1
                continue
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = peft_model(input_ids=input_ids, attention_mask=mask,
                                 use_cache=False, output_hidden_states=True)
                hidden_full = torch.cat(
                    [out.hidden_states[t] for t in config.taps], dim=-1
                ).float()
            hidden = hidden_full[mask.bool()]
            teacher_flat = torch.cat(
                [t[: int(n)] for t, n in zip(kept, mask.sum(1).tolist())], dim=0
            ).to(device)

            # 损失必须在 autocast 之外算：CKA 走 float32，余弦要稳定喵
            loss, parts = distill_loss(hidden, teacher_flat, head, stats, config)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params + [head], config.grad_clip)
            opt.step()
            sched.step()
            step += 1
            del out, hidden_full, hidden, teacher_flat, loss

            if step % config.log_every == 0 or step == 1:
                record = {"step": step, **parts, "sec": round(time.time() - started, 1)}
                result.history.append(record)
                print(f"step {step:>5} loss {parts['loss']:.6f} cka {parts['cka']:.6f} "
                      f"cos {parts['cos_loss']:.6f} col {parts['collapse']:.4f} "
                      f"{time.time() - started:.1f}s", flush=True)

            if config.eval_every and step % config.eval_every == 0:
                cos = evaluate_cosine(peft_model, tokenizer, reader, texts, val_idx,
                                      stats, head, config.taps, device, config.batch_tokens)
                print(f"  验证 step {step} 逐 token 余弦 {cos:.5f}", flush=True)
                result.history.append({"step": step, "val_cosine": cos})
                if cos > best:
                    best = cos
                    peft_model.save_pretrained(str(out_dir / "lora_best"))
                    torch.save(
                        {"head": head.detach().cpu(),
                         "mean_in": stats["mean_in"].cpu(),
                         "std_in": stats["std_in"].cpu(),
                         "mean_out": stats["mean_out"].cpu(),
                         "std_out": stats["std_out"].cpu(),
                         "val_cosine": cos},
                        out_dir / "head_best.pt",
                    )
                    print(f"  最优更新 -> {cos:.5f}", flush=True)

            if config.save_every and step % config.save_every == 0:
                save_checkpoint(peft_model, head, opt, sched, step, out_dir)

    result.steps = step
    peft_model.save_pretrained(str(out_dir / "lora"))
    torch.save(
        {
            "head": head.detach().cpu(),
            "mean_in": stats["mean_in"].cpu(),
            "std_in": stats["std_in"].cpu(),
            "mean_out": stats["mean_out"].cpu(),
            "std_out": stats["std_out"].cpu(),
        },
        out_dir / "head.pt",
    )
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
    adapter_path: Path,
    config: FinetuneConfig,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """训练前自检：LoRA 零初始化与损失实现是否真的等价喵。

    输入: 未挂 LoRA 的主干、分词器、样本、教师特征、Ridge 解、配置
    输出: 含 `lora_max_diff`、`cka`、`cos_loss` 的字典
    预期:
        - `lora_max_diff` 必须为 0.0：挂 LoRA 后 6 抽头拼接输出与原始模型逐位相同；
        - `cos_loss` 与 `cka` 应与训练日志 step 1 一致，因为那一步用的就是
          这组初始参数（LoRA 的 B 为零、head 为 Ridge 解）喵。
    """
    device = torch.device(device)
    adapter = load_file(str(adapter_path))
    stats = {
        "mean_in": adapter["mean_in"].to(device, torch.float32),
        "std_in": adapter["std_in"].to(device, torch.float32),
        "mean_out": adapter["mean_out"].to(device, torch.float32),
        "std_out": adapter["std_out"].to(device, torch.float32),
    }
    head = adapter["weight"].to(device, torch.float32)

    input_ids, mask, kept = make_batch(tokenizer, texts, teachers)
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    kept = [t.to(device) for t in kept]

    base = model(input_ids=input_ids, attention_mask=mask,
                 use_cache=False, output_hidden_states=True)
    base_h = torch.cat([base.hidden_states[t] for t in config.taps], dim=-1).float()
    del base

    peft_model, trainable, total = build_lora(model, config)
    peft_model.eval()
    lora_out = peft_model(input_ids=input_ids, attention_mask=mask,
                          use_cache=False, output_hidden_states=True)
    lora_h = torch.cat([lora_out.hidden_states[t] for t in config.taps], dim=-1).float()
    max_diff = float((lora_h - base_h).abs().max())

    hidden = base_h[mask.bool()]
    teacher_flat = torch.cat(
        [t[: int(n)] for t, n in zip(kept, mask.sum(1).tolist())], dim=0
    )
    _, parts = distill_loss(hidden, teacher_flat, head, stats, config)
    parts["lora_max_diff"] = max_diff
    parts["trainable"] = float(trainable)
    parts["total"] = float(total)
    return parts


@torch.inference_mode()
def reference_numbers(
    model,
    tokenizer,
    texts: list[str],
    teachers: list[np.ndarray],
    adapter_path: Path,
    config: FinetuneConfig,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """用不经过训练循环的路径复算 cos 与 CKA，作独立对照喵。

    输入: 未挂 LoRA 的主干、样本、教师特征、Ridge 解、配置
    输出: `{"cos_loss": ..., "cka": ...}`
    预期: 与 `self_check` 对应值差在 1e-4 以内；对不上说明批次对齐
          或标准化口径不一致喵。

    Notes:
        cos 用 `F.cosine_similarity` 手算，CKA 用 numpy 按定义重写一遍，
        不复用 `cka.py`，两边同时错的可能性很低喵。
    """
    device = torch.device(device)
    adapter = load_file(str(adapter_path))
    stats = {
        "mean_in": adapter["mean_in"].to(device, torch.float32),
        "std_in": adapter["std_in"].to(device, torch.float32),
        "mean_out": adapter["mean_out"].to(device, torch.float32),
        "std_out": adapter["std_out"].to(device, torch.float32),
    }
    head = adapter["weight"].to(device, torch.float32)

    input_ids, mask, kept = make_batch(tokenizer, texts, teachers)
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    kept = [t.to(device) for t in kept]

    out = model(input_ids=input_ids, attention_mask=mask,
                use_cache=False, output_hidden_states=True)
    h = torch.cat([out.hidden_states[t] for t in config.taps], dim=-1).float()[mask.bool()]
    t = torch.cat([x[: int(n)] for x, n in zip(kept, mask.sum(1).tolist())], dim=0)

    h_std = (h - stats["mean_in"]) / stats["std_in"]
    t_std = (t - stats["mean_out"]) / stats["std_out"]
    pred = h_std @ head * stats["std_out"] + stats["mean_out"]

    cos = torch.nn.functional.cosine_similarity(pred, t, dim=-1)
    cos_loss = float(1.0 - cos.mean())

    xs = h_std.detach().cpu().numpy().astype(np.float64)
    ys = t_std.detach().cpu().numpy().astype(np.float64)
    xs = xs - xs.mean(axis=0, keepdims=True)
    ys = ys - ys.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(ys.T @ xs, "fro")
    den = np.sqrt(np.linalg.norm(xs.T @ xs, "fro") * np.linalg.norm(ys.T @ ys, "fro"))
    return {"cos_loss": cos_loss, "cka": float(1.0 - cross / den)}


def save_checkpoint(peft_model, head, opt, sched, step: int, out_dir: Path) -> None:
    """存一份断点，含 LoRA、线性头、优化器与调度器状态喵。"""
    path = out_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "head": head.detach().cpu(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
        },
        path,
    )
    peft_model.save_pretrained(str(out_dir / f"lora_step_{step:06d}"))
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
    parser.add_argument("--adapter", type=Path, required=True, help="Ridge 解，作线性头初值")
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
    parser.add_argument("--val-tokens", type=int, default=40000,
                        help="验证集最多用多少 token")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="每多少步算一次验证余弦；0 表示不评估")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0,
                        help="每多少步存一次断点；0 表示只在最优点与结束时存")
    parser.add_argument("--merge", action="store_true",
                        help="把 LoRA 合并进主干并写到 --out/merged 喵")
    parser.add_argument("--lora", default="lora",
                        help="合并哪个子目录的 LoRA（默认 lora；验证最优点在 lora_best）")
    parser.add_argument("--check", action="store_true",
                        help="只做训练前自检：LoRA 零初始化 + 两条独立路径复算损失喵")
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

    train_tokens = sum(lengths[i] for i in train_idx)
    # 步数按真实批数算：短句多的一轮批数远多于「token/预算」，不能拿理论值估喵
    plan = build_length_batches(
        train_idx, lengths, args.batch_tokens, np.random.default_rng(0)
    )
    steps = args.steps or max(1, len(plan) * max(1, args.epochs))
    print(f"训练 {len(train_idx)} 条 / {train_tokens:,} token；"
          f"验证 {len(val_idx)} 条 / {used:,} token；"
          f"每轮 {len(plan)} 批，共 {steps} 步 x {args.batch_tokens} token", flush=True)

    reader = PromptFeatureReader(features, lengths, starts, d_out)
    enc = StudentEncoder(args.student, device=args.device, dtype=torch.bfloat16)
    tokenizer = MossTokenizer.from_file(args.tokenizer)

    config = FinetuneConfig(
        taps=taps,
        steps=steps,
        batch_tokens=args.batch_tokens,
        eval_every=args.eval_every,
        lora_r=args.lora_r,
        lr_lora=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
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
        ref = reference_numbers(
            enc.model, tokenizer, [texts[i] for i in pick], teachers,
            args.adapter, config, args.device
        )
        chk = self_check(
            enc.model, tokenizer, [texts[i] for i in pick], teachers,
            args.adapter, config, args.device
        )
        print(f"独立复算   cos_loss {ref['cos_loss']:.6f}   cka {ref['cka']:.6f}", flush=True)
        print(f"训练路径   cos_loss {chk['cos_loss']:.6f}   cka {chk['cka']:.6f}", flush=True)
        print(f"LoRA 零初始化最大差 {chk['lora_max_diff']:.3e}   "
              f"可训练 {int(chk['trainable']):,} / {int(chk['total']):,}", flush=True)
        d_cos = abs(ref["cos_loss"] - chk["cos_loss"])
        d_cka = abs(ref["cka"] - chk["cka"])
        ok = chk["lora_max_diff"] == 0.0 and d_cos < 1e-4 and d_cka < 1e-4
        print(f"自检{'通过' if ok else '失败'}  cos 差 {d_cos:.2e}  cka 差 {d_cka:.2e}", flush=True)
        reader.close()
        return 0 if ok else 1
    result = train_finetune(
        enc.model, tokenizer, reader, texts, train_idx, val_idx,
        args.adapter, config, args.out, args.device
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
