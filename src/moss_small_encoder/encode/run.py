"""把 prompts.jsonl 编码成 teacher / student 特征（两遍处理 + 断点续训）。

两遍而不是同时加载，是因为 8 GB 卡装不下「teacher 量化权重 3.4 GB + int8 词表
1.0 GB + student 1.7 GB + 批激活」的余量：

1. **teacher 遍**：只装 9B，批预算 4096，写 `teacher.raw`；
2. **student 遍**：卸载 9B 后只装 0.8B，批预算 16384，写 `student.raw`；
3. 两遍都按同一份「token 数降序」顺序逐个 prompt 写，所以两个文件逐行对齐；
4. 最后补 `meta.json` 与 `sink.npy`，格式由 `train/store.FeatureStore` 读取。

断点续训：每写完一整批就把 `*.progress.json` 落盘，里面记着已完成条数、
已写 token 数、文件字节偏移和累计 sink。中断后重跑会先 truncate 到那个
字节偏移再追加，所以不会出现半条 prompt 的脏数据。
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..tokenizer import MAX_TEXT_TOKENS, MossTokenizer
from .student import StudentEncoder
from .teacher import TeacherEncoder

TEACHER_CKPT = Path("models/qwen3.5_9b_w4a8_gptq.safetensors")
TEACHER_CFG = Path("models/qwen3.5-9b/config.json")
STUDENT_DIR = Path("models/qwen3.5-0.8b")
TOKENIZER = Path("models/tokenizer/tokenizer.json")

# MOSS 的 pad id（text_encoder.py:38 special_tokens={"pad": 248044}）
PAD_ID = 248044

# student 的隐状态维度。写进 meta，避免读的时候再猜。
STUDENT_DIM = 1024

# 分配器保留量超过这个值就把空闲块还给驱动。
# 依据: teacher 的真实峰值保留量实测 6.84 GB（T=3235），8 GB 卡留约 1 GB 给上下文和桌面；
# 分辨率越小批之间越碎，跑上百批后碎片会把保留量推过 8 GB，
# 随后被 WDDM 驱逐到共享显存，整批前向都在走 PCIe（实测从 2774 token/s 掉到约 700）。
RESERVED_LIMIT_BYTES = 7 * 1024**3


def load_prompts(
    path: Path,
    limit: int | None,
    max_tokens: int,
    seed: int = 20261027,
) -> list[dict]:
    """读 prompts.jsonl、按 token 数降序批内分桶，再整体洗成随机序。

    Args:
        path: 数据集文件。
        limit: 只取前多少条（按文件顺序），None 表示全部。
        max_tokens: 超过这个 token 数的样本丢弃。
        seed: 洗牌种子，固定值保证重跑编码结果一致。

    Returns:
        样本列表，整体已随机打乱（写入 raw 的行序 = 训练时的读序）。

    Notes:
        排序与洗牌都在这里做一次，两遍编码复用同一份顺序，保证
        teacher.raw 与 student.raw 逐行对齐。流程：先按长度降序
        （同批样本长度接近，减少 padding 浪费），再整表洗牌——
        两个操作都不影响行内对齐，只决定文件的物理行序。
        洗过的顺序让训练端纯顺序读就等于随机 SGD，无需任何下标机关。
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["n_tokens"] > max_tokens:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    rows.sort(key=lambda row: row["n_tokens"], reverse=True)
    np.random.default_rng(seed).shuffle(rows)
    return rows


def make_batches(
    rows: list[dict], budget: int, max_prompts: int, max_aspect: float = 3.0
) -> list[list[dict]]:
    """按 token 预算切批。

    Args:
        rows: 已排好序的样本。
        budget: 每批最多多少 token（按真实 token 计）。
        max_prompts: 每批最多多少条。
        max_aspect: 批内最长/最短长度比的上限。补齐到批内最长长度后，
            短样本会带来等量的无效计算，所以长度悬殊时宁可提前关批。
            默认 3.0 由实测选定：放宽到 4.0~6.0 补齐会变多，收紧到 1.6~2.0
            批数（因而固定开销）会翻倍，总耗时在 3.0 附近最低。

    Returns:
        批次列表。

    Notes:
        不在这里排序：调用方给的顺序就是写入顺序，两遍必须一致。
        单条超过预算的样本自己独占一批（长度已 ≤ 4096）。
        长度离散度这一条很关键：实测不限制时，4096 token 的批会被补齐到 16000+ token，
        等于四分之三的算力白烧，激活值也按补齐后的量算，直接顶穿 8 GB 显存。
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    longest = 0
    shortest = 0
    for row in rows:
        length = row["n_tokens"]
        # 双向判断：新样本比批内最长的短太多、或比最短的长太多，都会让补齐变浪费。
        # 只看一个方向不够——调用方给的是洗牌后的顺序，长样本可能后到。
        if current and (
            used + length > budget
            or len(current) >= max_prompts
            or longest > length * max_aspect
            or length > shortest * max_aspect
        ):
            batches.append(current)
            current, used, longest, shortest = [], 0, 0, 0
        if not current:
            shortest = length
        current.append(row)
        used += length
        longest = max(longest, length)
        shortest = min(shortest, length)
    if current:
        batches.append(current)
    return batches


def pad_batch(
    rows: list[dict], tokenizer: MossTokenizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """把一批样本补成矩形（右 padding）。

    Args:
        rows: 批内样本。
        tokenizer: MOSS 分词器。
        device: 目标设备。

    Returns:
        `(input_ids, attention_mask, lengths)`。

    Notes:
        causal mask 下真实 token 看不到后面的 pad，按 lengths 截取即可得到
        与逐条编码一致的结果。pad id 必须与生成侧一致（248044）。
    """
    encoded = [tokenizer.encode(row["text"]) for row in rows]
    lengths = [len(ids) for ids in encoded]
    width = max(lengths)
    input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, ids in enumerate(encoded):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, : len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device), lengths


class _RawWriter:
    """按 prompt 追加写 fp16 裸二进制，支持从断点截断后续写。

    Args:
        path: 输出文件。
        dim: 特征维度。
        resume: 上次的进度字典（含 `bytes` / `done` / `tokens`），None 表示从头写。

    Notes:
        重跑时先 truncate 到记录中的字节偏移，再以追加模式打开：
        中断时可能写了半批，truncate 把它们丢掉，保证文件只含完整的 prompt。
    """

    def __init__(self, path: Path, dim: int, resume: dict | None = None) -> None:
        self.path = Path(path)
        self.dim = dim
        if resume and self.path.exists():
            with self.path.open("r+b") as handle:
                handle.truncate(int(resume["bytes"]))
            self._handle = self.path.open("ab")
            self.n_tokens = int(resume["tokens"])
            self.n_prompts = int(resume["done"])
        else:
            self._handle = self.path.open("wb")
            self.n_tokens = 0
            self.n_prompts = 0

    @property
    def offset(self) -> int:
        """当前写入位置（字节）。"""
        return self._handle.tell()

    def add(self, features: np.ndarray) -> None:
        """写入一个 prompt 的特征。

        Args:
            features: `[tokens, dim]`，任意浮点类型，内部转 fp16。

        Raises:
            ValueError: 维度不符。
        """
        block = np.ascontiguousarray(features, dtype=np.float16)
        if block.shape[1] != self.dim:
            raise ValueError(f"维度不符 {block.shape}，期望 (*, {self.dim})")
        self._handle.write(block.tobytes())
        self.n_tokens += int(block.shape[0])
        self.n_prompts += 1

    def flush(self) -> None:
        """把缓冲刷到磁盘（写进度前必须调用）。"""
        self._handle.flush()

    def close(self) -> None:
        """关闭文件。"""
        self._handle.close()

    def __enter__(self) -> "_RawWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _read_progress(path: Path) -> dict | None:
    """读进度文件。

    Args:
        path: `*.progress.json`。

    Returns:
        字典，不存在或损坏时返回 None（那就是从头跑）。
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None


def _write_progress(path: Path, payload: dict) -> None:
    """原子写进度文件。

    Args:
        path: 目标路径。
        payload: 要写的字典。

    Notes:
        先写 `.tmp` 再 replace，避免断电时留下半个 JSON。

        Windows 上杀毒软件、搜索索引或别的进程（包括用户随手打开文件查看）
        可能短暂持有目标文件，`os.replace` 会抛 `PermissionError`。
        这里重试几次；仍然失败就放弃这一次更新——进度是每批都写的，
        少写一次只是续跑粒度差一批，不该让整个编码进程崩掉。
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    temporary.unlink(missing_ok=True)
    print(f"[warn] 进度文件被占用，跳过本次写入: {path}", flush=True)


def _run_teacher_pass(
    rows: list[dict],
    tokenizer: MossTokenizer,
    out_dir: Path,
    device: torch.device,
    budget: int,
    max_prompts: int,
) -> tuple[list[int], np.ndarray, float]:
    """teacher 遍：装 9B，逐个 prompt 写 `teacher.raw`。

    Args:
        rows: 排好序的全部样本。
        tokenizer: 分词器。
        out_dir: 输出目录。
        device: 计算设备。
        budget: 每批 token 预算。
        max_prompts: 每批最多条数。

    Returns:
        `(lengths, sink 向量, 耗时秒)`。

    Notes:
        token 0 是 attention sink（方向恒定、范数极大），不参与拟合；
        这里顺带统计所有 prompt 的 token 0 均值，供推理时替换。
        进度文件每批更新，中断重跑会从整批边界继续。
    """
    progress_path = out_dir / "teacher.progress.json"
    sink_path = out_dir / "teacher.sink_sum.npy"
    resume = _read_progress(progress_path)
    if resume and resume.get("total") != len(rows):
        # 样本集变了，进度作废，从头来。
        resume = None
        progress_path.unlink(missing_ok=True)
        sink_path.unlink(missing_ok=True)

    start = int(resume["done"]) if resume else 0
    lengths: list[int] = list(resume["lengths"]) if resume else []
    sink_sum = (
        np.load(sink_path).astype(np.float64)
        if resume and sink_path.exists()
        else np.zeros(4096, dtype=np.float64)
    )
    sink_count = int(resume["sink_count"]) if resume else 0

    remaining = rows[start:]
    batches = make_batches(remaining, budget, max_prompts)
    total = sum(row["n_tokens"] for row in rows)
    started = time.time()
    teacher = TeacherEncoder(TEACHER_CKPT, TEACHER_CFG, device=device)
    print(
        f"[teacher] 就位 {time.time() - started:.1f}s，显存 {torch.cuda.memory_allocated()/1e9:.2f} GB"
        + (f"，从第 {start} 条续跑" if start else ""),
        flush=True,
    )

    done = int(sum(lengths))
    written = start
    with _RawWriter(out_dir / "teacher.raw", teacher.hidden_size, resume) as writer:
        for index, batch in enumerate(batches, 1):
            input_ids, attention_mask, batch_lengths = pad_batch(batch, tokenizer, device)
            with torch.inference_mode():
                hidden = teacher.encode(input_ids, attention_mask)
            # 直接转 fp16 再回 CPU：落盘就是 fp16，先转 fp32 等于把 PCIe 传输量翻倍。
            hidden_np = hidden.to(torch.float16).cpu().numpy()
            for row_index, length in enumerate(batch_lengths):
                row = hidden_np[row_index, :length]
                writer.add(row)
                sink_sum += row[0].astype(np.float64)
                sink_count += 1
                lengths.append(length)
            written += len(batch)
            done += sum(batch_lengths)

            writer.flush()
            np.save(sink_path, sink_sum)
            _write_progress(
                progress_path,
                {
                    "done": written,
                    "tokens": writer.n_tokens,
                    "bytes": writer.offset,
                    "lengths": lengths,
                    "sink_count": sink_count,
                    "total": len(rows),
                },
            )

            # 批长度差异大会留下无法复用的空闲块，累积后被驱动驱逐到共享显存。
            # 超过阈值就把空闲块还回去；下一批重新申请是毫秒级开销，远小于溢出后的 PCIe 往返。
            if torch.cuda.memory_reserved() > RESERVED_LIMIT_BYTES:
                torch.cuda.empty_cache()

            if index % 20 == 0 or index == len(batches):
                elapsed = time.time() - started
                rate = (done - int(sum(lengths[:start]))) / elapsed if elapsed else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                print(
                    f"[teacher] 批 {index}/{len(batches)}  {done:,}/{total:,} token  "
                    f"{rate:.0f} token/s  剩余 {eta:.1f} 分",
                    flush=True,
                )

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    sink = (sink_sum / max(sink_count, 1)).astype(np.float32)
    return lengths, sink, time.time() - started


def _run_student_pass(
    rows: list[dict],
    tokenizer: MossTokenizer,
    out_dir: Path,
    device: torch.device,
    budget: int,
    max_prompts: int,
    taps: tuple[int, ...],
) -> float:
    """student 遍：装 0.8B，逐个 prompt 写 `student.raw`。

    Args:
        rows: 排好序的全部样本（与 teacher 遍同一顺序）。
        tokenizer: 分词器。
        out_dir: 输出目录。
        device: 计算设备。
        budget: 每批 token 预算。
        max_prompts: 每批最多条数。
        taps: 取哪些层，按顺序拼接成一条更宽的向量。`-1` 表示最后一层
            （已过 final norm）。取同一 token 的不同深度，不做跨 token 混合。

    Returns:
        耗时秒。
    """
    progress_path = out_dir / "student.progress.json"
    resume = _read_progress(progress_path)
    if resume and resume.get("total") != len(rows):
        resume = None
        progress_path.unlink(missing_ok=True)

    start = int(resume["done"]) if resume else 0
    remaining = rows[start:]
    batches = make_batches(remaining, budget, max_prompts)
    total = sum(row["n_tokens"] for row in rows)
    started = time.time()
    student = StudentEncoder(STUDENT_DIR, device=device)
    print(
        f"[student] 就位 {time.time() - started:.1f}s，显存 {torch.cuda.memory_allocated()/1e9:.2f} GB"
        + (f"，从第 {start} 条续跑" if start else ""),
        flush=True,
    )

    written = start
    done = int(resume["tokens"]) if resume else 0
    with _RawWriter(
        out_dir / "student.raw", student.hidden_size * len(taps), resume
    ) as writer:
        for index, batch in enumerate(batches, 1):
            input_ids, attention_mask, batch_lengths = pad_batch(batch, tokenizer, device)
            with torch.inference_mode():
                hidden = student.encode(input_ids, attention_mask, taps=taps)
            # 沿特征维拼接多层：第 i 个 token 得到 各层隐状态 的串接。
            hidden_np = torch.cat(
                [hidden[t].to(torch.float16) for t in taps], dim=-1
            ).cpu().numpy()
            for row_index, length in enumerate(batch_lengths):
                writer.add(hidden_np[row_index, :length])
            written += len(batch)
            done += sum(batch_lengths)

            writer.flush()
            _write_progress(
                progress_path,
                {
                    "done": written,
                    "tokens": writer.n_tokens,
                    "bytes": writer.offset,
                    "total": len(rows),
                },
            )

            if index % 20 == 0 or index == len(batches):
                elapsed = time.time() - started
                rate = (done - (int(resume["tokens"]) if resume else 0)) / elapsed if elapsed else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                print(
                    f"[student] 批 {index}/{len(batches)}  {done:,}/{total:,} token  "
                    f"{rate:.0f} token/s  剩余 {eta:.1f} 分",
                    flush=True,
                )

    del student
    gc.collect()
    torch.cuda.empty_cache()
    return time.time() - started


def auto_student_budget(taps: tuple[int, ...], device: torch.device) -> int:
    """按当前可用显存反推学生批的 token 预算。

    Args:
        taps: 抽头层数（每 token 要存这么多份 1024 维特征）。
        device: 计算设备。

    Returns:
        建议的批 token 数，取 2 的幂，夹在 8192..131072 之间。

    Notes:
        学生模型只有 1.7 GB（9B 教师是 4.95 GB），沿用教师的 16384 会让每批
        的矩阵太窄、固定开销占比过高。这里按「可用显存 - 3 GB（模型+工作区）」
        反推：每 token 存 len(taps) 份 fp16 特征，再乘 2 的临时量余量。
        显存不够时退回到保守值，不会把机器顶爆。
    """
    free, _ = torch.cuda.mem_get_info(device)
    headroom = max(int(free) - 3 * 1024**3, 256 * 1024**2)
    per_token = max(len(taps), 1) * STUDENT_DIM * 2 * 2
    budget = headroom // per_token
    budget = min(max(budget, 8192), 65536)
    # 向下取到 2 的幂，避免每批形状都不同造成的分配器碎片
    return 1 << (budget.bit_length() - 1)


def main(argv: list[str] | None = None) -> int:
    """入口。

    Args:
        argv: 命令行参数，None 表示用 sys.argv。

    Returns:
        进程退出码。

    Notes:
        产物目录含 `teacher.raw` / `student.raw` / `meta.json` / `sink.npy`；
        中途中断时 `meta.json` 不存在，但两个 `.raw` 可以续跑；
        全部完成后进度文件会被删掉，避免下次误判为「已完成」。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("dataset/prompts.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("dataset/features/v1"))
    parser.add_argument("--teacher-budget", type=int, default=4096, help="teacher 批 token 预算")
    parser.add_argument("--student-budget", type=int, default=0, help="student 批 token 预算；0 表示按显存自动推断")
    parser.add_argument("--teacher-max-prompts", type=int, default=64, help="teacher 每批最多条数")
    parser.add_argument("--student-max-prompts", type=int, default=256, help="student 每批最多条数")
    parser.add_argument("--limit", type=int, default=None, help="只编码前 N 条")
    parser.add_argument("--tap", type=int, default=-1, help="student 取哪一层（单层）")
    parser.add_argument(
        "--taps",
        type=str,
        default=None,
        help="逗号分隔的多层抽头，例如 '8,16,-1'；给出时按顺序拼接，d_in = 层数 x 1024",
    )
    args = parser.parse_args(argv)

    # 单 tap 保持向后兼容；给了 --taps 就以它为准。
    taps = tuple(int(x) for x in args.taps.split(",")) if args.taps else (args.tap,)

    rows = load_prompts(args.prompts, args.limit, MAX_TEXT_TOKENS)
    if not rows:
        raise SystemExit("没有可用样本")
    args.out.mkdir(parents=True, exist_ok=True)

    total_tokens = sum(row["n_tokens"] for row in rows)
    print(f"样本 {len(rows)} 条 / {total_tokens:,} token", flush=True)

    device = torch.device("cuda")
    tokenizer = MossTokenizer.from_file(TOKENIZER)

    # 学生比教师小得多，固定预算会浪费算力；0 表示按当前可用显存反推。
    student_budget = args.student_budget or auto_student_budget(taps, device)
    print(f"[student] 批预算 {student_budget} token（taps={len(taps)}）", flush=True)

    lengths, sink, teacher_seconds = _run_teacher_pass(
        rows,
        tokenizer,
        args.out,
        device,
        args.teacher_budget,
        args.teacher_max_prompts,
    )
    student_seconds = _run_student_pass(
        rows,
        tokenizer,
        args.out,
        device,
        args.student_budget,
        args.student_max_prompts,
        taps,
    )

    np.save(args.out / "sink.npy", sink)
    meta = {
        "d_in": STUDENT_DIM * len(taps),
        "d_out": int(sink.shape[0]),
        "dtype": "float16",
        "n_prompts": len(rows),
        "n_tokens": int(sum(lengths)),
        "lengths": lengths,
        "prompt_ids": [row["id"] for row in rows],
        "sink_excluded": True,
        "tap": taps[-1],
        "taps": list(taps),
        "seconds": {
            "teacher": round(teacher_seconds, 1),
            "student": round(student_seconds, 1),
        },
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name in ("teacher.progress.json", "student.progress.json", "teacher.sink_sum.npy"):
        (args.out / name).unlink(missing_ok=True)

    print(
        f"完成：{len(rows)} 条 / {sum(lengths):,} token -> {args.out}"
        f"（teacher {teacher_seconds/60:.1f} 分，student {student_seconds/60:.1f} 分）",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
