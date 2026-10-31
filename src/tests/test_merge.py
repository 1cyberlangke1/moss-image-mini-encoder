"""语料合并模块的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from moss_small_encoder import merge


def _write(path: Path, records: list[dict]) -> None:
    """把记录写成 prompts.jsonl喵。

    输入: 目标路径、记录列表
    输出: 无
    预期: 每行一个 JSON，带 n_tokens / bucket / id 喵
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(record_id: str, bucket: str, tokens: int = 5) -> dict:
    """造一条最小纪录喵。"""
    return {
        "id": record_id,
        "bucket": bucket,
        "source": "test",
        "source_id": record_id,
        "kind": "positive",
        "lang": "zh",
        "nsfw": False,
        "n_tokens": tokens,
        "split": "train",
        "text_sha1": record_id,
        "text": record_id,
    }


def test_split_of_is_deterministic() -> None:
    """同一 id 反复调用必须落在同一侧喵。"""
    ratios = (0.7, 0.15, 0.15)
    for record_id in ("a", "b", "c", "character:1:000"):
        assert merge.split_of(record_id, ratios) == merge.split_of(record_id, ratios)


def test_split_of_respects_ratio_order() -> None:
    """比例全给 train 时任何 id 都该进 train喵。"""
    assert merge.split_of("whatever", (1.0, 0.0, 0.0)) == "train"


def test_rate_of_is_stable_and_bounded() -> None:
    """`rate_of` 必须落在 [0,1) 且对同一 id 稳定喵。"""
    for record_id in ("x", "y", "z", "artists:danbooru:1:a000"):
        value = merge.rate_of(record_id)
        assert value == merge.rate_of(record_id)
        assert 0.0 <= value < 1.0


def test_default_puts_everything_in_train(tmp_path: Path, monkeypatch) -> None:
    """不给 --split 时全部记录都该进 train喵。"""
    kb = tmp_path / "kb.jsonl"
    hard = tmp_path / "hard.jsonl"
    replay = tmp_path / "replay.jsonl"
    _write(kb, [_record(f"kb:{i}", "characters") for i in range(10)])
    _write(hard, [_record("hard:1", "hard_cases")])
    _write(replay, [_record("replay:1", "en_tag")])
    monkeypatch.setattr(
        merge,
        "SOURCES",
        (("kb", kb), ("hard_cases", hard), ("replay", replay)),
    )

    out = tmp_path / "corpus"
    assert merge.main(["--out", str(out)]) == 0

    rows = [json.loads(line) for line in (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert {row["split"] for row in rows} == {"train"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["by_split"] == {"train": 12}
    assert manifest["rows"] == 12
    assert manifest["tokens"] == 60


def test_explicit_split_produces_all_three(tmp_path: Path, monkeypatch) -> None:
    """显式给比例时三个桶都该出现，且总数不变喵。"""
    kb = tmp_path / "kb.jsonl"
    _write(kb, [_record(f"kb:{i}", "characters") for i in range(200)])
    monkeypatch.setattr(merge, "SOURCES", (("kb", kb),))

    out = tmp_path / "corpus"
    assert merge.main(["--out", str(out), "--split", "0.7,0.15,0.15"]) == 0

    rows = [json.loads(line) for line in (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    assert len(rows) == 200
    assert set(counts) == {"train", "val", "test"}


def test_duplicate_ids_are_dropped(tmp_path: Path, monkeypatch) -> None:
    """同一个 id 出现两次时只保留第一条喵。"""
    kb = tmp_path / "kb.jsonl"
    _write(kb, [_record("dup:1", "layout"), _record("dup:1", "layout")])
    monkeypatch.setattr(merge, "SOURCES", (("kb", kb),))

    out = tmp_path / "corpus"
    assert merge.main(["--out", str(out)]) == 0
    rows = (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_missing_source_is_skipped(tmp_path: Path, monkeypatch) -> None:
    """来源文件缺失时跳过而不是崩喵。"""
    shared = tmp_path / "kb.jsonl"
    _write(shared, [_record("kb:1", "characters")])
    monkeypatch.setattr(
        merge,
        "SOURCES",
        (("kb", shared), ("hard_cases", tmp_path / "missing.jsonl")),
    )

    out = tmp_path / "corpus"
    assert merge.main(["--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
