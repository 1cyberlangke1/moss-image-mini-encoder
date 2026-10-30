"""构建流程里纯逻辑部分的单测（不碰网络、不碰模型）。"""

from __future__ import annotations

from collections import Counter

import pytest

from moss_small_encoder import build


RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class TestSplitOf:
    def test_deterministic(self):
        assert build.split_of("layout:abc", RATIOS) == build.split_of("layout:abc", RATIOS)

    def test_ratios_roughly_match(self):
        counts = Counter(build.split_of(f"id:{i}", RATIOS) for i in range(20000))
        assert 0.68 < counts["train"] / 20000 < 0.72
        assert 0.13 < counts["val"] / 20000 < 0.17
        assert 0.13 < counts["test"] / 20000 < 0.17

    def test_all_ids_are_assigned(self):
        counts = Counter(build.split_of(f"id:{i}", RATIOS) for i in range(500))
        assert set(counts) == {"train", "val", "test"}


class TestReservoir:
    @staticmethod
    def _items(n):
        return ({**{"source": "s", "source_id": str(i)}, "i": i} for i in range(n))

    def test_respects_quota(self):
        stats = Counter()
        kept = build.reservoir(self._items(1000), 10, seed=1, stats=stats)
        assert len(kept) == 10
        assert stats["candidates"] == 1000
        assert stats["over_quota"] == 990

    def test_keeps_everything_under_quota(self):
        stats = Counter()
        kept = build.reservoir(self._items(5), 10, seed=1, stats=stats)
        assert len(kept) == 5
        assert stats["over_quota"] == 0

    def test_deterministic(self):
        a = build.reservoir(self._items(500), 20, seed=7, stats=Counter())
        b = build.reservoir(self._items(500), 20, seed=7, stats=Counter())
        assert [x["i"] for x in a] == [x["i"] for x in b]

    def test_seed_does_not_matter(self):
        # 抽样只依赖 source_id 哈希，换种子不该改变结果（种子只管合成桶的内容）。
        a = build.reservoir(self._items(300), 20, seed=1, stats=Counter())
        b = build.reservoir(self._items(300), 20, seed=999, stats=Counter())
        assert [x["i"] for x in a] == [x["i"] for x in b]

    def test_larger_quota_is_a_superset(self):
        # 配额 20 -> 40 时，原来那 20 条必须原样保留在前面，只往后追加。
        small = build.reservoir(self._items(500), 20, seed=7, stats=Counter())
        large = build.reservoir(self._items(500), 40, seed=7, stats=Counter())
        assert [x["i"] for x in small] == [x["i"] for x in large[:20]]

    def test_selection_is_not_prefix_of_input(self):
        # 抽样必须真的挑，不能退化成「取前 N 条」。
        kept = build.reservoir(self._items(500), 20, seed=7, stats=Counter())
        assert [x["i"] for x in kept] != list(range(20))


class TestDropReason:
    CFG = {"min_chars": 12, "max_chars": 200}

    @pytest.mark.parametrize(
        ("source", "kind", "text", "expected"),
        [
            ("jtatman", "positive", "short", "too_short"),
            ("jtatman", "positive", "a long enough prompt about a cat", None),
            ("jtatman", "positive", "x" * 201, "too_long_chars"),
            (
                "shuixian",
                "positive",
                "在保持原图完全不变的前提下进行编辑，添加一些元素",
                "edit_instruction",
            ),
            (
                "shuixian",
                "positive",
                "一个穿着汉服的女性站在竹林里，柔和侧光，胶片质感",
                None,
            ),
            (
                "goku",
                "positive",
                "Redraw the attached image in a clumsy style please",
                "edit_instruction",
            ),
            ("goku", "positive", "A woman standing in a bamboo forest at dawn", "not_zh"),
            ("n714", "positive", "黑发, 长发, 汉服, 柔和侧光", None),
            ("synth_layout", "positive", '{"summary":{"caption":"x"}}', None),
        ],
    )
    def test_reasons(self, source, kind, text, expected):
        assert build.drop_reason(source, kind, text, self.CFG) == expected

    def test_official_is_not_checked_for_language(self):
        # 官方样例是基准，不能被「非中文」这类规则筛掉。
        text = "A wide view of a refrigerator beside a kitchen counter."
        assert build.drop_reason("moss_official", "positive", text, self.CFG) is None


class TestTokenSummary:
    def test_empty(self):
        assert build._token_summary([]) == {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}

    def test_values(self):
        summary = build._token_summary([10, 20, 30, 40])
        assert summary["mean"] == 25.0
        assert summary["median"] == 25.0
        assert summary["max"] == 40.0
        assert 35.5 <= summary["p95"] <= 40.0


class TestPackageVersions:
    def test_contains_python_and_tokenizers(self):
        versions = build._package_versions()
        assert "python" in versions
        assert "tokenizers" in versions
