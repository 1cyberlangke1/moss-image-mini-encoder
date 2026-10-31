"""packed 编码路径的测试。

覆盖点：
- `pack_batch` 拼平后的偏移与长度严格对应，顺序不变
- `make_batches` 在 packed 模式下不再按长度离散度关批
- `_RawWriter` 在没有续跑记录时拒绝覆盖已有文件
- `_verify_packed_against_stored` 能检出前后不一致，也能在一致时放行
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from moss_small_encoder.encode.run import (
    _RawWriter,
    _verify_packed_against_stored,
    make_batches,
    pack_batch,
)


class _FakeTokenizer:
    """按文本返回预设 token 序列的分词器替身。"""

    def __init__(self, table: dict[str, list[int]]) -> None:
        self.table = table

    def encode(self, text: str) -> list[int]:
        """返回该文本预设的 token 序列。"""
        return list(self.table[text])


class _FakeTeacher:
    """固定输出的教师替身，用来验证对拍逻辑本身。"""

    def __init__(self, hidden_size: int, payload: np.ndarray) -> None:
        self.hidden_size = hidden_size
        self.payload = payload

    def encode_packed(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """返回与 `payload` 完全一致的隐状态。"""
        assert input_ids.shape == (1, self.payload.shape[0])
        assert int(cu_seqlens[-1]) == self.payload.shape[0]
        return torch.from_numpy(self.payload.astype(np.float32)).unsqueeze(0)


def _rows(lengths: list[int]) -> list[dict]:
    """按给定长度造样本列表（id 递增，便于校验顺序）。

    Args:
        lengths: 每条样本的 token 数。

    Returns:
        与 `load_prompts` 同形状的行列表。
    """
    return [{"id": f"p{i:04d}", "text": f"t{i}", "n_tokens": n} for i, n in enumerate(lengths)]


def test_pack_batch_offsets_match_lengths() -> None:
    """拼平后按 cu_seqlens 切片能还原每条样本。

    输入: 三条长度 3 / 1 / 4 的样本
    输出: `input_ids` 是首尾相接的 8 个 token，`cu_seqlens` 为 [0, 3, 4, 8]
    预期: 偏移单调递增、末项等于总长，切片结果与逐条编码一致
    """
    tokenizer = _FakeTokenizer({"t0": [1, 2, 3], "t1": [4], "t2": [5, 6, 7, 8]})
    rows = _rows([3, 1, 4])
    input_ids, cu_seqlens, lengths = pack_batch(rows, tokenizer, torch.device("cpu"))

    assert lengths == [3, 1, 4]
    assert input_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert cu_seqlens.tolist() == [0, 3, 4, 8]

    flat = input_ids[0].tolist()
    bounds = cu_seqlens.tolist()
    restored = [flat[bounds[i] : bounds[i + 1]] for i in range(len(lengths))]
    assert restored == [[1, 2, 3], [4], [5, 6, 7, 8]]


def test_pack_batch_rejects_empty_sample() -> None:
    """空样本必须报错，否则 cu_seqlens 会出现重复值。

    输入: 一条长度为 0 的样本
    输出: ValueError
    预期: 下游无法靠重复的 cu_seqlens 区分空段，所以入口就拒绝
    """
    tokenizer = _FakeTokenizer({"t0": [], "t1": [1, 2]})
    with pytest.raises(ValueError):
        pack_batch(_rows([0, 2]), tokenizer, torch.device("cpu"))


def test_make_batches_packed_ignores_aspect() -> None:
    """packed 不补齐，长度悬殊也不该提前关批。

    输入: 长度 1 与 1000 交替的 8 条样本，预算 4096
    输出: packed 版批次更少，且摊平顺序与原列表一致
    预期: 不 packed 时长度离散度会拆成多批；packed 时按预算填满
    """
    rows = _rows([1, 1000] * 4)
    loose = make_batches(rows, budget=4096, max_prompts=64)
    packed = make_batches(rows, budget=4096, max_prompts=64, packed=True)

    assert len(packed) <= len(loose)
    assert [row["id"] for batch in packed for row in batch] == [row["id"] for row in rows]
    assert sum(len(batch) for batch in packed) == len(rows)


def test_raw_writer_refuses_to_overwrite_existing(tmp_path) -> None:
    """没有续跑记录时不得覆盖已有特征。

    输入: 一个已有 32 字节的 teacher.raw，resume=None
    输出: FileExistsError，文件内容原样保留
    预期: 续跑指纹误判时宁可停下，也不能把已有特征清零
    """
    path = tmp_path / "teacher.raw"
    path.write_bytes(b"\x00" * 32)

    with pytest.raises(FileExistsError):
        _RawWriter(path, dim=4, resume=None)
    assert path.stat().st_size == 32


def test_raw_writer_appends_from_resume(tmp_path) -> None:
    """给了续跑记录就截断到记录位置后继续追加。

    输入: 已写 2 个 token 的 dim=2 文件，记录 tokens=2 / bytes=8
    输出: 新写入的只有 1 个 token
    预期: 文件总长 12 字节，前 8 字节不变
    """
    path = tmp_path / "teacher.raw"
    first = np.arange(4, dtype=np.float16).reshape(2, 2)
    path.write_bytes(first.tobytes())

    with _RawWriter(path, dim=2, resume={"bytes": 8, "tokens": 2, "done": 1}) as writer:
        writer.add(np.array([[9.0, 9.0]], dtype=np.float16))

    payload = np.frombuffer(path.read_bytes(), dtype=np.float16)
    assert payload.tolist() == [0.0, 1.0, 2.0, 3.0, 9.0, 9.0]


def test_require_packed_deps_rejects_missing_kernel(monkeypatch) -> None:
    """缺 causal-conv1d 时必须直接报错，不能静默退化。

    输入: 把 `is_causal_conv1d_available` 打成 False
    输出: ImportError，信息里点名 causal-conv1d
    预期: 退化版 conv1d 不看 seq_idx，会跨段串味且不报错，所以入口就要拦
    """
    from moss_small_encoder.encode import teacher as teacher_module

    monkeypatch.setattr(teacher_module, "is_causal_conv1d_available", lambda: False)
    with pytest.raises(ImportError, match="causal-conv1d"):
        teacher_module._require_packed_deps()


def test_teacher_pass_aborts_and_backs_up_on_invalid_resume(tmp_path) -> None:
    """进度校验不通过且盘上已有特征时，必须停下来而不是删进度。

    输入: 一个非空的 teacher.raw，加一份指纹对不上的进度文件
    输出: SystemExit；teacher.raw 原样保留；进度被复制成 .bak
    预期: 这条路上不能删 progress.json 和 sink_sum.npy，
          否则恢复所需的偏移和 sink 累计会一起消失
    """
    from moss_small_encoder.encode.run import _run_teacher_pass

    rows = [{"id": "p0", "text": "t0", "n_tokens": 2}]
    raw = tmp_path / "teacher.raw"
    raw.write_bytes(b"\x00" * 16)
    (tmp_path / "teacher.sink_sum.npy").write_bytes(b"\x00")
    progress = {
        "done": 1,
        "tokens": 2,
        "bytes": 16,
        "lengths": [2],
        "sink_count": 1,
        "total": 1,
        "fingerprint": "deadbeef",
        "settings": {},
    }
    (tmp_path / "teacher.progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        _run_teacher_pass(rows, None, tmp_path, torch.device("cpu"), 2048, 64)

    assert raw.read_bytes() == b"\x00" * 16
    assert (tmp_path / "teacher.progress.json.bak").exists()
    assert (tmp_path / "teacher.progress.json").exists()
    assert (tmp_path / "teacher.sink_sum.npy").exists()


def _stored_and_teacher(tmp_path, scale: float):
    """造一份已写入特征与一个返回同内容的教师替身。

    Args:
        tmp_path: 临时目录。
        scale: 教师输出相对已写入特征的倍率，-1 表示完全反向。

    Returns:
        `(路径, 教师, 样本, 长度)`。
    """
    hidden, total = 4, 7
    rng = np.random.default_rng(20261027)
    stored = rng.normal(size=(total, hidden)).astype(np.float16)
    path = tmp_path / "teacher.raw"
    path.write_bytes(stored.tobytes())

    teacher = _FakeTeacher(hidden, stored.astype(np.float64) * scale)
    tokenizer = _FakeTokenizer({"t0": [1, 2, 3, 4], "t1": [5, 6, 7]})
    rows = [{"id": "p0", "text": "t0", "n_tokens": 4}, {"id": "p1", "text": "t1", "n_tokens": 3}]
    return path, teacher, tokenizer, rows, [4, 3]


def test_verify_packed_passes_when_identical(tmp_path) -> None:
    """前后一致时对拍通过，余弦接近 1。

    输入: 已写入特征与教师输出完全相同
    输出: 余弦均值与最小值都约等于 1
    预期: 跳过 token 0 后仍有 4 个 token 参与比较
    """
    path, teacher, tokenizer, rows, lengths = _stored_and_teacher(tmp_path, 1.0)
    metrics = _verify_packed_against_stored(
        teacher, tokenizer, rows, lengths, path, torch.device("cpu")
    )

    assert metrics["samples"] == 2
    assert metrics["tokens"] == (4 - 1) + (3 - 1)
    assert metrics["mean"] > 0.9999
    assert metrics["token_mean"] > 0.9999
    assert metrics["token_min"] > 0.9999
    assert metrics["max_abs"] < 1e-6


def test_verify_packed_aborts_when_diverged(tmp_path) -> None:
    """前后不一致时必须中止，而不是继续写盘。

    输入: 教师输出是已写入特征的相反数
    输出: SystemExit，异常信息里带 metrics
    预期: 这道闸在打开写入器之前触发，文件不会被改动
    """
    path, teacher, tokenizer, rows, lengths = _stored_and_teacher(tmp_path, -1.0)
    before = path.read_bytes()

    with pytest.raises(SystemExit):
        _verify_packed_against_stored(
            teacher, tokenizer, rows, lengths, path, torch.device("cpu")
        )
    assert path.read_bytes() == before
