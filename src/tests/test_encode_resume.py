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
