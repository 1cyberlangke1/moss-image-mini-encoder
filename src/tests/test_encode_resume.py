"""续跑指纹校验的单元测试。"""

from __future__ import annotations

from pathlib import Path

from moss_small_encoder.encode.run import _resume_valid, _rows_fingerprint


def test_fingerprint_is_stable_and_order_sensitive() -> None:
    """同样顺序同样 id 得同一指纹；换顺序或换 id 必须变。

    输入: 三组样本列表
    输出: 指纹字符串
    预期: 前两组相同，后两组不同喵
    """
    rows_a = [{"id": "x"}, {"id": "y"}]
    rows_b = [{"id": "x"}, {"id": "y"}]
    rows_c = [{"id": "y"}, {"id": "x"}]
    rows_d = [{"id": "x"}, {"id": "z"}]

    assert _rows_fingerprint(rows_a) == _rows_fingerprint(rows_b)
    assert _rows_fingerprint(rows_a) != _rows_fingerprint(rows_c)
    assert _rows_fingerprint(rows_a) != _rows_fingerprint(rows_d)


def test_resume_valid_accepts_matching_state(tmp_path: Path) -> None:
    """指纹、设置、文件大小都对得上时应允许续跑喵。"""
    raw = tmp_path / "student.raw"
    raw.write_bytes(b"\x00" * 64)
    progress = {"bytes": 64, "fingerprint": "abc", "settings": {"taps": [1, -1]}}

    assert _resume_valid(progress, fingerprint="abc", settings={"taps": [1, -1]}, raw_path=raw)


def test_resume_valid_accepts_appended_rows(tmp_path: Path) -> None:
    """只在末尾追加样本时，指纹变了也要允许续跑喵。

    输入: 旧进度记了 3 条长度、raw 已写满；新样本集是这 3 条再加 2 条
    输出: `_resume_valid` 的布尔值
    预期: True，因为旧 raw 的每一行仍然对得上喵
    """
    raw = tmp_path / "teacher.raw"
    raw.write_bytes(b"\x00" * 64)
    progress = {
        "bytes": 64,
        "fingerprint": "old",
        "settings": {},
        "lengths": [5, 7, 9],
    }

    assert _resume_valid(
        progress,
        fingerprint="new",
        settings={},
        raw_path=raw,
        row_lengths=[5, 7, 9, 11, 13],
    )


def test_resume_valid_rejects_changed_prefix(tmp_path: Path) -> None:
    """前缀不一致说明旧 raw 的行已经错位，必须拒绝续跑喵。

    输入: 旧进度记了 [5, 7, 9]，新样本集是 [5, 8, 9, 11]
    输出: `_resume_valid` 的布尔值
    预期: False，第 2 条长度就对不上喵
    """
    raw = tmp_path / "teacher.raw"
    raw.write_bytes(b"\x00" * 64)
    progress = {
        "bytes": 64,
        "fingerprint": "old",
        "settings": {},
        "lengths": [5, 7, 9],
    }

    assert not _resume_valid(
        progress,
        fingerprint="new",
        settings={},
        raw_path=raw,
        row_lengths=[5, 8, 9, 11],
    )


def test_resume_valid_rejects_shrunk_rows(tmp_path: Path) -> None:
    """新样本集比旧进度还短时不是追加，必须拒绝喵。"""
    raw = tmp_path / "teacher.raw"
    raw.write_bytes(b"\x00" * 64)
    progress = {
        "bytes": 64,
        "fingerprint": "old",
        "settings": {},
        "lengths": [5, 7, 9],
    }

    assert not _resume_valid(
        progress,
        fingerprint="new",
        settings={},
        raw_path=raw,
        row_lengths=[5, 7],
    )


def test_resume_valid_rejects_changed_settings(tmp_path: Path) -> None:
    """改了 taps 之后旧进度必须作废，否则会写出混步长文件喵。"""
    raw = tmp_path / "student.raw"
    raw.write_bytes(b"\x00" * 64)
    progress = {"bytes": 64, "fingerprint": "abc", "settings": {"taps": [1, -1]}}

    assert not _resume_valid(
        progress, fingerprint="abc", settings={"taps": [1, 2, -1]}, raw_path=raw
    )


def test_resume_valid_rejects_missing_or_short_raw(tmp_path: Path) -> None:
    """裸文件被删或比记录的短，说明数据不完整，不能续写喵。"""
    raw = tmp_path / "student.raw"
    progress = {"bytes": 64, "fingerprint": "abc", "settings": {}}

    assert not _resume_valid(progress, fingerprint="abc", settings={}, raw_path=raw)

    raw.write_bytes(b"\x00" * 16)
    assert not _resume_valid(progress, fingerprint="abc", settings={}, raw_path=raw)


def test_resume_valid_rejects_missing_sidecar(tmp_path: Path) -> None:
    """sink 累计文件丢了就不能续跑，否则分母非零而分子为零喵。"""
    raw = tmp_path / "teacher.raw"
    raw.write_bytes(b"\x00" * 64)
    sidecar = tmp_path / "teacher.sink_sum.npy"
    progress = {"bytes": 64, "fingerprint": "abc", "settings": {}}

    assert not _resume_valid(
        progress, fingerprint="abc", settings={}, raw_path=raw, sidecar=sidecar
    )
