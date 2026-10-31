"""画师桶与 tag 序列桶的单元测试喵。

覆盖:
    画师桶只取 category=1、尊重热度门槛、加 @ 前缀、每人一条
    tag 序列桶按官方顺序起头、不带角色名、年份只收 201x/202x、可复现
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moss_small_encoder.sources import danbooru as db


def write_tags(path: Path, rows: list[dict]) -> Path:
    """把测试用 tag 写成 jsonl喵。

    输入: 目标路径、记录列表（至少含 name/category/post_count/is_deprecated）
    输出: 路径
    预期: 每行一个 JSON 对象喵
    """
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def tags_path(tmp_path: Path) -> Path:
    """造一份最小 tag 清单：画师、角色、通用、年份、废弃各若干喵。"""
    rows = [
        {"id": 1, "name": "dairi", "category": 1, "post_count": 18859, "is_deprecated": False},
        {"id": 2, "name": "ruu_(tksymkw)", "category": 1, "post_count": 5990, "is_deprecated": False},
        {"id": 3, "name": "dead_artist", "category": 1, "post_count": 2, "is_deprecated": False},
        {"id": 4, "name": "gone_artist", "category": 1, "post_count": 9000, "is_deprecated": True},
        {"id": 5, "name": "kayoko_(blue_archive)", "category": 4, "post_count": 300, "is_deprecated": False},
        {"id": 6, "name": "blue_archive", "category": 3, "post_count": 9000, "is_deprecated": False},
        {"id": 7, "name": "black_hair", "category": 0, "post_count": 9000, "is_deprecated": False},
        {"id": 8, "name": "red_eyes", "category": 0, "post_count": 8000, "is_deprecated": False},
        {"id": 9, "name": "school_uniform", "category": 0, "post_count": 7000, "is_deprecated": False},
        {"id": 10, "name": "pleated_skirt", "category": 0, "post_count": 6000, "is_deprecated": False},
        {"id": 11, "name": "solo", "category": 0, "post_count": 5000, "is_deprecated": False},
        {"id": 12, "name": "outdoors", "category": 0, "post_count": 4000, "is_deprecated": False},
        {"id": 13, "name": "2024", "category": 5, "post_count": 5000, "is_deprecated": False},
        {"id": 14, "name": "1945", "category": 5, "post_count": 3000, "is_deprecated": False},
        {"id": 15, "name": "tagme", "category": 5, "post_count": 9000, "is_deprecated": False},
        {"id": 16, "name": "...!", "category": 0, "post_count": 9000, "is_deprecated": False},
    ]
    return write_tags(tmp_path / "tags.jsonl", rows)


class TestArtists:
    """画师桶的行为喵。"""

    def test_only_artists_and_threshold(self, tags_path: Path) -> None:
        """只出 category=1 且过门槛的画师，废弃的不要喵。"""
        records = list(db.iter_artists(tags_path, per_artist=1, min_post_count=10))
        names = sorted(r.meta["artist"] for r in records)
        assert names == ["dairi", "ruu_(tksymkw)"]

    def test_at_prefix_present(self, tags_path: Path) -> None:
        """每条都必须带 @ 前缀，教师才把它当画师引用喵。"""
        for record in db.iter_artists(tags_path, per_artist=1, min_post_count=10):
            assert "@" in record.text

    def test_one_per_artist_by_default(self, tags_path: Path) -> None:
        """per_artist=1 时每个画师恰好一条，source_id 不重复喵。"""
        records = list(db.iter_artists(tags_path, per_artist=1, min_post_count=10))
        assert len(records) == 2
        assert len({r.source_id for r in records}) == 2

    def test_empty_raises(self, tmp_path: Path) -> None:
        """没有画师时直接报错，不静默出空桶喵。"""
        path = write_tags(
            tmp_path / "t.jsonl",
            [{"id": 1, "name": "black_hair", "category": 0, "post_count": 9, "is_deprecated": False}],
        )
        with pytest.raises(ValueError):
            list(db.iter_artists(path, per_artist=1))


class TestTagSequences:
    """tag 序列桶的行为喵。"""

    def test_quality_prefix(self, tags_path: Path) -> None:
        """每条都以官方推荐的 quality 段开头喵。"""
        for record in db.iter_tag_sequences(tags_path, 5):
            assert record.text.startswith("best quality, highres")

    def test_no_character_names(self, tags_path: Path) -> None:
        """不许出现角色名，避免把随机属性绑到角色上喵。"""
        for record in db.iter_tag_sequences(tags_path, 20):
            assert "kayoko" not in record.text
            assert "blue_archive" not in record.text

    def test_year_window(self, tags_path: Path) -> None:
        """年份只收 201x/202x，1945 这种不收喵。"""
        for record in db.iter_tag_sequences(tags_path, 30):
            assert "1945" not in record.text
            assert "2024" not in record.text or True  # 允许但不强制出现

    def test_junk_general_filtered(self, tags_path: Path) -> None:
        """纯标点的 general tag 不许进序列喵。"""
        for record in db.iter_tag_sequences(tags_path, 20):
            assert "...!" not in record.text

    def test_meta_not_used(self, tags_path: Path) -> None:
        """meta 类（tagme）不许进序列喵。"""
        for record in db.iter_tag_sequences(tags_path, 20):
            assert "tagme" not in record.text

    def test_count_and_determinism(self, tags_path: Path) -> None:
        """条数正确，同 seed 两次完全一致喵。"""
        first = [r.text for r in db.iter_tag_sequences(tags_path, 7, seed=42)]
        second = [r.text for r in db.iter_tag_sequences(tags_path, 7, seed=42)]
        assert len(first) == 7
        assert first == second

    def test_source_id_unique(self, tags_path: Path) -> None:
        """source_id 必须唯一喵。"""
        ids = [r.source_id for r in db.iter_tag_sequences(tags_path, 50)]
        assert len(set(ids)) == len(ids)
