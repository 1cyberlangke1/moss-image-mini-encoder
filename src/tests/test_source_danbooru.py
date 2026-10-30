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
