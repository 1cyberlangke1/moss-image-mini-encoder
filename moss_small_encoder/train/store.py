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
