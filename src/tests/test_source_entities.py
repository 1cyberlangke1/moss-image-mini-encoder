"""Wikidata 实体展开成提示词的单元测试喵。"""

from __future__ import annotations

import json
from pathlib import Path

from moss_small_encoder.sources import entities


def _write(path: Path, rows: list[dict]) -> Path:
    """把实体记录写成 jsonl，返回路径喵。"""
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _entity(qid: str, name: str, sitelinks: int, root: str = "品牌") -> dict:
    return {"qid": qid, "name": name, "sitelinks": sitelinks, "type": "Q431289", "root": root}


def test_make_variants_is_deterministic_and_unique() -> None:
    first = entities.make_variants("蜜雪冰城", 40, 20261027)
    second = entities.make_variants("蜜雪冰城", 40, 20261027)

    assert first == second
    assert len(first) == 40
    assert len(set(first)) == 40
    assert all("蜜雪冰城" in text for text in first)


def test_make_variants_contains_no_appearance_words() -> None:
    """槽位里不能出现外观词，否则就变成「描述 → 向量」的循环喵。"""
    forbidden = ("发", "瞳", "红色", "白色", "黑色", "裙子", "帽子", "围巾", "hair", "eyes")
    for text in entities.make_variants("蜜雪冰城", 200, 20261027):
        for word in forbidden:
            assert word not in text


def test_make_variants_caps_at_total_combinations() -> None:
    variants = entities.make_variants("蜜雪冰城", 10**6, 20261027)
    # 6 个模板只用部分槽位，去重后上限 137：
    # 场景 8 + (场景×视角) 40 + (场景×天气) 48 + (视角×天气) 30 + 视角 5 + 天气 6喵
    assert len(variants) == 137
    assert len(set(variants)) == 137


def test_make_variants_uses_english_for_ascii_name() -> None:
    variants = entities.make_variants("GameStop", 10, 20261027)
    assert all("GameStop" in text for text in variants)
    assert any("logo" in text or "storefront" in text for text in variants)


def test_load_entities_keeps_low_frequency_and_dedupes_by_name(tmp_path: Path) -> None:
    path = _write(tmp_path / "e.jsonl", [
        _entity("Q1", "蜜雪冰城", 25),
        _entity("Q2", "蜜雪冰城", 3),
        _entity("Q3", "雪王", 0),
    ])

    rows = entities.load_entities(path)

    assert [r["qid"] for r in rows] == ["Q1", "Q3"]
    assert rows[-1]["sitelinks"] == 0


def test_iter_records_ids_are_unique(tmp_path: Path) -> None:
    path = _write(tmp_path / "e.jsonl", [
        _entity("Q1", "蜜雪冰城", 25),
        _entity("Q2", "雪王", 1),
    ])

    records = list(entities.iter_records(path, per_entity=12))

    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)) == 24
    assert all(r["bucket"] == "entity" for r in records)
    assert {r["lang"] for r in records} == {"zh"}
