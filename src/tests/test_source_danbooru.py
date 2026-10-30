"""Danbooru tag 源的单元测试：清单读取、属性词表清洗、提示词拼装喵。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moss_small_encoder.sources import danbooru


def _write_tags(path: Path, rows: list[dict]) -> Path:
    """把给定记录写成 jsonl，返回路径喵。"""
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _tag(tid: int, name: str, count: int, category: int, deprecated: bool = False) -> dict:
    """造一条 tag 记录，字段跟下载脚本写出来的一致喵。"""
    return {
        "id": tid,
        "name": name,
        "post_count": count,
        "category": category,
        "is_deprecated": deprecated,
    }


def test_load_tags_drops_deprecated_and_sorts_by_count(tmp_path: Path) -> None:
    path = _write_tags(tmp_path / "tags.jsonl", [
        _tag(1, "rare_tag", 5, 0),
        _tag(2, "old_tag", 999, 0, deprecated=True),
        _tag(3, "popular_tag", 5000, 0),
    ])

    tags = danbooru.load_tags(path)

    assert [t["name"] for t in tags] == ["popular_tag", "rare_tag"]


def test_load_attr_pool_skips_cosplay_and_parenthesised(tmp_path: Path) -> None:
    (tmp_path / "feature.csv").write_text(
        "tag,count\nlong_hair,100\nblue_eyes,90\nlong_hair,80\n", encoding="utf-8"
    )
    (tmp_path / "attire.csv").write_text(
        "tag,count\nlucina_(fire_emblem)_(cosplay),50\nblack_skirt,40\n", encoding="utf-8"
    )

    pool = danbooru.load_attr_pool(tmp_path)

    assert pool == ["long_hair", "blue_eyes", "black_skirt"]


def test_load_attr_pool_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert danbooru.load_attr_pool(tmp_path / "nope") == []


def test_iter_records_rotates_three_formats_and_is_deterministic(tmp_path: Path) -> None:
    path = _write_tags(tmp_path / "tags.jsonl", [
        _tag(10, "hatsune_miku", 130000, 4),
        _tag(11, "kayoko_(blue_archive)", 9000, 4),
        _tag(12, "vocaloid", 200000, 3),
        _tag(13, "long_hair", 500000, 0),
        _tag(14, "black_skirt", 90000, 0),
        _tag(15, "smile", 400000, 0),
    ])
    attrs = tmp_path / "attrs"
    attrs.mkdir()
    (attrs / "feature.csv").write_text("tag,count\nlong_hair,500000\n", encoding="utf-8")
    (attrs / "attire.csv").write_text("tag,count\nblack_skirt,90000\n", encoding="utf-8")

    first = list(danbooru.iter_records(path, count=6, seed=7, attr_dir=attrs))
    second = list(danbooru.iter_records(path, count=6, seed=7, attr_dir=attrs))

    assert [r.text for r in first] == [r.text for r in second]
    assert len(first) == 6
    # 三种格式轮换：第 0 条纯下划线 tag 串，第 2 条是自然句喵
    assert "_" in first[0].text.split(",")[0]
    assert first[2].text.startswith("a picture of ")
    assert all(r.source == "danbooru" for r in first)
    assert all(r.text for r in first)


def test_iter_records_rejects_empty_vocab(tmp_path: Path) -> None:
    path = _write_tags(tmp_path / "tags.jsonl", [_tag(1, "solo_tag", 1, 0)])

    with pytest.raises(ValueError):
        list(danbooru.iter_records(path, count=1, attr_dir=None))


class TestDownloadPaging:
    """分页硬限制的回归测试：不空转、不跳过、不重复写喵。"""

    @staticmethod
    def _row(tid: int) -> dict:
        """造一条 tag 记录喵。"""
        return {
            "id": tid,
            "name": f"tag_{tid}",
            "post_count": 1,
            "category": 0,
            "is_deprecated": False,
        }

    def test_last_page_bisects_instead_of_walking(self, monkeypatch, tmp_path):
        """区间共 7 页时，二分应直接定位到 7，而不是逐页翻喵。"""
        calls: list[int] = []

        def fake(session, page, lo=0, hi=0):
            calls.append(page)
            return [self._row(page)] if page <= 7 else []

        monkeypatch.setattr(danbooru, "_fetch_page", fake)
        assert danbooru._last_page(None, 0, 1000) == 7
        assert len(calls) < 20

    def test_empty_range_returns_zero(self, monkeypatch):
        """区间为空时页数为 0，不发起后续请求喵。"""
        monkeypatch.setattr(danbooru, "_fetch_page", lambda *a, **k: [])
        assert danbooru._last_page(None, 0, 1000) == 0

    def test_download_writes_all_and_does_not_overfetch(self, monkeypatch, tmp_path):
        """3 页区间只写 3 条，且总请求次数远小于逐页翻到底喵。"""
        calls: list[int] = []

        def fake(session, page, lo=0, hi=0):
            calls.append(page)
            if page <= 3:
                return [self._row(page)]
            return []

        monkeypatch.setattr(danbooru, "_fetch_page", fake)
        monkeypatch.setattr(danbooru, "_opener", lambda proxy: None)

        out = tmp_path / "tags.jsonl"
        n = danbooru.download_tags(
            out, proxy="", ranges=[(0, 1000)], sleep=0, report_every=0
        )

        assert n == 3
        assert len(calls) < 40
        assert len(out.read_text(encoding="utf-8").splitlines()) == 3

    def test_download_raises_instead_of_skipping_a_page(self, monkeypatch, tmp_path):
        """某页连续失败必须抛错，绝不静默跳过喵。"""
        def fake(session, page, lo=0, hi=0):
            if page == 2:
                raise OSError("boom")
            return [self._row(page)] if page <= 3 else []

        monkeypatch.setattr(danbooru, "_fetch_page", fake)
        monkeypatch.setattr(danbooru, "_opener", lambda proxy: None)
        monkeypatch.setattr(danbooru.time, "sleep", lambda _: None)

        with pytest.raises(RuntimeError):
            danbooru.download_tags(
                tmp_path / "tags.jsonl", proxy="", ranges=[(0, 1000)],
                sleep=0, report_every=0, retries=2,
            )

    def test_rerun_skips_existing_ids(self, monkeypatch, tmp_path):
        """重跑时已存在的 id 不重复写喵。"""
        def fake(session, page, lo=0, hi=0):
            return [self._row(page)] if page <= 3 else []

        monkeypatch.setattr(danbooru, "_fetch_page", fake)
        monkeypatch.setattr(danbooru, "_opener", lambda proxy: None)

        out = tmp_path / "tags.jsonl"
        first = danbooru.download_tags(out, proxy="", ranges=[(0, 1000)],
                                       sleep=0, report_every=0)
        second = danbooru.download_tags(out, proxy="", ranges=[(0, 1000)],
                                        sleep=0, report_every=0)

        assert first == second == 3
        assert len(out.read_text(encoding="utf-8").splitlines()) == 3
