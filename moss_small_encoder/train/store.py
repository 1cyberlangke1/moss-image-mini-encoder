"""特征落盘与读取。

teacher / student 的特征按 token 展平存成裸二进制（float16），另存每个 prompt
的 token 数。这样：

- 追加写：编码阶段按 prompt 顺序写，不需要预先知道总量；
- 零拷贝读：训练时 `np.memmap` 打开，随机取 batch 不会把 40 GB 读进内存；
- 分组清楚：每个 token 属于哪个 prompt 由 token 数前缀和算出，
  早停必须按 prompt 分组，否则同一 prompt 的 token 会泄漏进验证集。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPE = np.float16


class FeatureWriter:
    """按 prompt 追加写特征。

    Args:
        directory: 输出目录，会创建 `student.raw` / `teacher.raw` / `meta.json`。
        d_in: student 维度。
        d_out: teacher 维度。

    Notes:
        **不要写入 token 0（attention sink）**。它的范数是普通 token 的几十倍，
        用 float16 存有溢出成 inf 的风险；它又本来就不参与拟合。
        推理时需要的那个常量向量用 `add_sink` 单独存一次。
    """

    def __init__(self, directory: Path, d_in: int, d_out: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.d_in = d_in
        self.d_out = d_out
        self._student = (self.directory / "student.raw").open("wb")
        self._teacher = (self.directory / "teacher.raw").open("wb")
        self.lengths: list[int] = []
        self.prompt_ids: list[str] = []

    def add(self, prompt_id: str, student: np.ndarray, teacher: np.ndarray) -> None:
        """追加一个 prompt 的特征。

        Args:
            prompt_id: 样本 id，写进 meta 便于回溯。
            student: `[tokens, d_in]`，会被转成 float16。
            teacher: `[tokens, d_out]`，同上。

        Raises:
            ValueError: 两个数组的 token 数不一致，或维度不符。

        Notes:
            两个数组的 token 数必须相等：投影是逐位置对齐的，
            长度不同说明编码阶段出了问题，必须当场报错而不是静默截断。
        """
        if student.shape[0] != teacher.shape[0]:
            raise ValueError(
                f"{prompt_id}: student {student.shape[0]} token != teacher {teacher.shape[0]} token"
            )
        if student.shape[1] != self.d_in or teacher.shape[1] != self.d_out:
            raise ValueError(
                f"{prompt_id}: 维度不符 {student.shape} / {teacher.shape}，"
                f"期望 (*, {self.d_in}) / (*, {self.d_out})"
            )
        self._student.write(np.ascontiguousarray(student, dtype=DTYPE).tobytes())
        self._teacher.write(np.ascontiguousarray(teacher, dtype=DTYPE).tobytes())
        self.lengths.append(int(student.shape[0]))
        self.prompt_ids.append(prompt_id)

    def close(self) -> None:
        """关闭文件并写 meta.json。

        Returns:
            无。
        """
        self._student.close()
        self._teacher.close()
        meta = {
            "d_in": self.d_in,
            "d_out": self.d_out,
            "dtype": np.dtype(DTYPE).name,
            "n_prompts": len(self.lengths),
            "n_tokens": int(sum(self.lengths)),
            "lengths": self.lengths,
            "prompt_ids": self.prompt_ids,
            "sink_excluded": True,
        }
        (self.directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def add_sink(self, sink: np.ndarray) -> None:
        """保存 teacher 的 sink 向量（所有 prompt 的 token 0 均值）。

        Args:
            sink: `[d_out]`，float32 也可以，内部存 float32。

        Returns:
            无。

        Notes:
            存 float32 而不是 float16：sink 是推理时要加回去的常量，
            它本身数值大，半精度会丢有效位。
        """
        np.save(self.directory / "sink.npy", np.asarray(sink, dtype=np.float32))

    def __enter__(self) -> "FeatureWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class FeatureStore:
    """只读特征视图。

    Attributes:
        d_in: student 维度。
        d_out: teacher 维度。
        lengths: 每个 prompt 的 token 数。
        prompt_ids: 每个 prompt 的 id。
        student: `[n_tokens, d_in]` 的内存映射。
        teacher: `[n_tokens, d_out]` 的内存映射。
    """

    d_in: int
    d_out: int
    lengths: np.ndarray
    prompt_ids: list[str]
    student: np.ndarray
    teacher: np.ndarray

    @property
    def n_tokens(self) -> int:
        """总 token 数。

        Returns:
            所有 prompt 的 token 数之和。
        """
        return int(self.lengths.sum())

    @property
    def n_prompts(self) -> int:
        """prompt 总数。

        Returns:
            `lengths` 的长度。
        """
        return int(self.lengths.shape[0])

    @classmethod
    def open(cls, directory: Path) -> "FeatureStore":
        """打开已写好的特征目录。

        Args:
            directory: 含 `meta.json` 与两个 `.raw` 的目录。

        Returns:
            FeatureStore 实例（文件以 memmap 打开，不占内存）。
        """
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        n_tokens = int(meta["n_tokens"])
        d_in, d_out = int(meta["d_in"]), int(meta["d_out"])
        dtype = np.dtype(meta.get("dtype", "float16"))
        student = np.memmap(
            directory / "student.raw", dtype=dtype, mode="r", shape=(n_tokens, d_in)
        )
        teacher = np.memmap(
            directory / "teacher.raw", dtype=dtype, mode="r", shape=(n_tokens, d_out)
        )
        return cls(
            d_in=d_in,
            d_out=d_out,
            lengths=np.asarray(meta["lengths"], dtype=np.int64),
            prompt_ids=list(meta["prompt_ids"]),
            student=student,
            teacher=teacher,
        )

    def groups(self) -> np.ndarray:
        """给每个 token 标上它所属的 prompt 序号。

        Returns:
            长度 `n_tokens` 的 int32 数组，值域 `[0, n_prompts)`。
        """
        return np.repeat(np.arange(self.n_prompts, dtype=np.int32), self.lengths)

    def split_by_prompt(self, ratios: dict[str, float]) -> dict[str, np.ndarray]:
        """按 prompt 分配 train/val/test，返回各集合的 prompt 下标。

        Args:
            ratios: `{"train": 0.7, "val": 0.15, "test": 0.15}`。

        Returns:
            划分名到 prompt 下标数组的映射。

        Notes:
            用 prompt id 的稳定哈希分配，保证同一份数据每次划到同一侧，
            且与 token 内容无关。
        """
        import hashlib

        keys = np.array(
            [
                int(hashlib.sha1(pid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
                for pid in self.prompt_ids
            ]
        )
        train_cut = ratios["train"]
        val_cut = train_cut + ratios["val"]
        return {
            "train": np.flatnonzero(keys < train_cut),
            "val": np.flatnonzero((keys >= train_cut) & (keys < val_cut)),
            "test": np.flatnonzero(keys >= val_cut),
        }

    def token_indices(self, prompt_indices: np.ndarray) -> np.ndarray:
        """把 prompt 下标展开成 token 下标。

        Args:
            prompt_indices: prompt 序号数组。

        Returns:
            这些 prompt 覆盖的全部 token 下标（按 prompt 顺序连续）。
        """
        starts = np.concatenate(([0], np.cumsum(self.lengths)))
        blocks = [
            np.arange(starts[i], starts[i + 1], dtype=np.int64) for i in prompt_indices
        ]
        if not blocks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(blocks)

    def token_spans(self, prompt_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把 prompt 下标折叠成连续 token 区间（升序、互不重叠）。

        Args:
            prompt_indices: prompt 序号数组（顺序不限）。

        Returns:
            `(starts, ends)`，第 i 个区间是 `[starts[i], ends[i])`。

        Notes:
            排序后的 prompt 下标在 token 轴上常形成连续段；折叠后
            区间数远少于 prompt 数，下游可以按区间顺序流式切片。
        """
        order = np.sort(np.asarray(prompt_indices, dtype=np.int64))
        starts = np.concatenate(([0], np.cumsum(self.lengths)))
        begin = starts[order]
        end = starts[order + 1]
        merged_s: list[int] = []
        merged_e: list[int] = []
        for s, e in zip(begin.tolist(), end.tolist()):
            if merged_s and s <= merged_e[-1]:
                merged_e[-1] = max(merged_e[-1], e)
            else:
                merged_s.append(s)
                merged_e.append(e)
        return (
            np.asarray(merged_s, dtype=np.int64),
            np.asarray(merged_e, dtype=np.int64),
        )


class PromptFeatureDataset:
    """标准 PyTorch Dataset，通过文件流式 seek/read 按 prompt 读取特征，零 memmap 膨胀。"""

    def __init__(self, directory: Path, prompt_indices: Sequence[int], drop_first: int = 1):
        import torch

        self.directory = Path(directory)
        meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        self.d_in = int(meta["d_in"])
        self.d_out = int(meta["d_out"])
        self.lengths = np.asarray(meta["lengths"], dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.lengths)))
        self.indices = np.asarray(prompt_indices, dtype=np.int64)
        self.drop_first = drop_first
        self.f_student = open(self.directory / "student.raw", "rb")
        self.f_teacher = open(self.directory / "teacher.raw", "rb")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        import torch

        p = int(self.indices[idx])
        L = int(self.lengths[p])
        s = int(self.starts[p])
        offset = min(self.drop_first, L)
        valid_L = L - offset
        if valid_L <= 0:
            return (
                torch.empty((0, self.d_in), dtype=torch.float16),
                torch.empty((0, self.d_out), dtype=torch.float16),
            )

        self.f_student.seek((s + offset) * self.d_in * 2)
        x = torch.frombuffer(
            bytearray(self.f_student.read(valid_L * self.d_in * 2)),
            dtype=torch.float16,
        ).view(valid_L, self.d_in)

        self.f_teacher.seek((s + offset) * self.d_out * 2)
        y = torch.frombuffer(
            bytearray(self.f_teacher.read(valid_L * self.d_out * 2)),
            dtype=torch.float16,
        ).view(valid_L, self.d_out)

        return x, y

    def close(self):
        self.f_student.close()
        self.f_teacher.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def collate_prompt_features(batch):
    """PyTorch DataLoader collate 函数：拼接 batch 内各 prompt 的 token。"""
    import torch

    xs = [b[0] for b in batch if b[0].shape[0] > 0]
    ys = [b[1] for b in batch if b[1].shape[0] > 0]
    if not xs:
        return torch.empty((0, 1024), dtype=torch.float16), torch.empty((0, 4096), dtype=torch.float16)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class StreamReader:
    """直接使用标准二进制文件流读取特征，不依赖 np.memmap，彻底杜绝操作系统工作集膨胀。"""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        self.d_in = int(meta["d_in"])
        self.d_out = int(meta["d_out"])
        self.n_tokens = int(meta["n_tokens"])
        self.lengths = np.asarray(meta["lengths"], dtype=np.int64)
        self.prompt_ids = list(meta["prompt_ids"])
        self.starts = np.concatenate(([0], np.cumsum(self.lengths)))
        self.f_student = open(self.directory / "student.raw", "rb")
        self.f_teacher = open(self.directory / "teacher.raw", "rb")

    def close(self):
        self.f_student.close()
        self.f_teacher.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def n_prompts(self) -> int:
        return int(self.lengths.shape[0])

    def split_by_prompt(self, ratios: dict[str, float]) -> dict[str, np.ndarray]:
        import hashlib

        keys = np.array(
            [
                int(hashlib.sha1(pid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
                for pid in self.prompt_ids
            ]
        )
        train_cut = ratios["train"]
        val_cut = train_cut + ratios["val"]
        return {
            "train": np.flatnonzero(keys < train_cut),
            "val": np.flatnonzero((keys >= train_cut) & (keys < val_cut)),
            "test": np.flatnonzero(keys >= val_cut),
        }

    def token_indices(self, prompt_indices: np.ndarray) -> np.ndarray:
        starts = np.concatenate(([0], np.cumsum(self.lengths)))
        blocks = [
            np.arange(starts[i], starts[i + 1], dtype=np.int64) for i in prompt_indices
        ]
        if not blocks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(blocks)

    def read_chunk_into(
        self,
        prompt_indices: Sequence[int],
        x_buf: np.ndarray,
        y_buf: np.ndarray,
        drop_first: int = 1,
        offset: int = 0,
    ) -> int:
        """批量读取多个 prompt 到预分配固定缓冲区中，在文件 seek 层面跳过 attention sink。

        严格保证按文件物理起始位置升序单调读取，严禁任何逆向/乱序寻道。

        连续 prompt 走「整段读 + 就地压缩」路径：只发一次 seek+read，拿到包含 sink 的
        原始区间后在内存里把每条 prompt 的首 token 抹掉。散装 prompt（不连续）退回逐条读。

        Args:
            prompt_indices: prompt 下标列表。
            x_buf: 学生特征缓冲区，形状 `[容量, d_in]`。
            y_buf: 教师特征缓冲区，形状 `[容量, d_out]`。
            drop_first: 每个 prompt 开头跳过的 token 数（默认 1，即跳过 sink）。
            offset: 缓冲区写入起始偏移（用于保留上一批未满 batch 的尾数）。

        Returns:
            实际写入缓冲区的有效 token 数量。

        Notes:
            压缩是「向前搬」的（目标位置恒小于源位置），所以可以在同一块缓冲区内原地做，
            不需要额外的临时数组；每条 prompt 的搬运量是 L-1 行，纯内存 memmove。
        """
        p_sorted = sorted(prompt_indices, key=lambda idx: self.starts[idx])
        if not p_sorted:
            return 0

        # 判断这些 prompt 在文件里是否首尾相接。训练/验证的 prompt 列表都是按
        # 物理位置连续取出的，所以这里几乎总是成立。
        contiguous = True
        for a, b in zip(p_sorted, p_sorted[1:]):
            if int(self.starts[a]) + int(self.lengths[a]) != int(self.starts[b]):
                contiguous = False
                break

        if not contiguous:
            # 稀疏 prompt：退回逐条 seek，保持语义不变。
            cur = offset
            for p in p_sorted:
                L = int(self.lengths[p])
                s = int(self.starts[p])
                off = min(drop_first, L)
                valid_L = L - off
                if valid_L <= 0:
                    continue
                self.f_student.seek((s + off) * self.d_in * 2)
                self.f_student.readinto(x_buf[cur : cur + valid_L])
                self.f_teacher.seek((s + off) * self.d_out * 2)
                self.f_teacher.readinto(y_buf[cur : cur + valid_L])
                cur += valid_L
            return cur - offset

        # 连续路径：整段一次读完（含各条 prompt 的 token 0），再就地压缩。
        span_start = int(self.starts[p_sorted[0]])
        span_end = int(self.starts[p_sorted[-1]] + self.lengths[p_sorted[-1]])
        span = span_end - span_start

        # 护栏：整段读要求缓冲区装得下 offset + span，否则 numpy 切片会被静默截断，
        # 读进来半截数据还照样返回，训练时表现为莫名其妙的 loss 异常。
        if offset + span > x_buf.shape[0] or offset + span > y_buf.shape[0]:
            raise ValueError(
                f"缓冲区不足：需要 {offset + span} 行，实际只有 "
                f"x={x_buf.shape[0]} / y={y_buf.shape[0]} 行"
            )

        self.f_student.seek(span_start * self.d_in * 2)
        self.f_student.readinto(x_buf[offset : offset + span])
        self.f_teacher.seek(span_start * self.d_out * 2)
        self.f_teacher.readinto(y_buf[offset : offset + span])

        read_pos = offset
        write_pos = offset
        for p in p_sorted:
            L = int(self.lengths[p])
            skip = min(drop_first, L)
            keep = L - skip
            read_pos += skip
            if keep > 0:
                if read_pos != write_pos:
                    x_buf[write_pos : write_pos + keep] = x_buf[read_pos : read_pos + keep]
                    y_buf[write_pos : write_pos + keep] = y_buf[read_pos : read_pos + keep]
                write_pos += keep
            read_pos += keep
        return write_pos - offset
