"""layout 合成与序列化的单测。

序列化格式必须与生成侧逐字一致，所以这里固定住：
键顺序、紧凑分隔符、中文不转义、bbox 取整方式。
"""

import json

import pytest

from moss_small_encoder import synth_layout


class TestCenteredBox:
    def test_matches_documented_example(self):
        # layout.md 的例子：1536x1024 画布上的 [176,224,800,864]
        # 应当得到 [-288,-592,352,32]。
        assert synth_layout.centered_box([176, 224, 800, 864], 1536, 1024) == [
            -288,
            -592,
            352,
            32,
        ]

    def test_full_canvas_is_symmetric(self):
        assert synth_layout.centered_box([0, 0, 1024, 1024], 1024, 1024) == [
            -512,
            -512,
            512,
            512,
        ]

    def test_fractional_edges_floor_and_ceil(self):
        # 左/上 floor，右/下 ceil：保证框只能变大不能变小，不会漏掉像素。
        assert synth_layout.centered_box([10.4, 20.7, 100.2, 200.9], 1024, 1024) == [
            20 - 512,
            10 - 512,
            201 - 512,
            101 - 512,
        ]

    @pytest.mark.parametrize(
        "box",
        [
            [-1, 0, 10, 10],
            [0, 0, 1025, 10],
            [10, 10, 10, 20],
            [0, 0, 10],
        ],
    )
    def test_rejects_bad_boxes(self, box):
        with pytest.raises(ValueError):
            synth_layout.centered_box(box, 1024, 1024)

    def test_rejects_canvas_not_multiple_of_16(self):
        with pytest.raises(ValueError):
            synth_layout.centered_box([0, 0, 10, 10], 1000, 1000)

    def test_rejects_degenerate_after_rounding(self):
        # 差 1e-10 的框在保留 9 位小数后会退化成零宽度，必须拒绝而不是写出空框。
        with pytest.raises(ValueError):
            synth_layout.centered_box([10.0, 10.0, 10.0000000001, 10.5], 1024, 1024)


class TestCompileDraft:
    def test_key_order_and_compaction(self):
        draft = {
            "width": 1024,
            "height": 1024,
            "bbox_coord_space": "abs",
            "bbox_format": "xyxy",
            "summary": {"caption": "poster"},
            "items": [{"label": "object", "caption": "a cup", "bbox": [0, 0, 512, 512]}],
        }
        out = synth_layout.compile_draft(draft)
        # 紧凑：没有空格；键顺序固定。
        assert " " not in out.replace("a cup", "")
        assert out.startswith('{"summary":{"caption":"poster"},"items":[')
        assert out.endswith(',"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}')

    def test_chinese_not_escaped(self):
        draft = {
            "width": 1024,
            "height": 1024,
            "bbox_coord_space": "abs",
            "bbox_format": "xyxy",
            "summary": {"caption": "极简海报"},
            "items": [{"label": "text", "caption": "标题", "text": "把日常，过成诗。", "bbox": [0, 0, 512, 512]}],
        }
        out = synth_layout.compile_draft(draft)
        assert "极简海报" in out
        assert "\\u" not in out

    def test_item_key_order_preserved(self):
        draft = {
            "width": 1024,
            "height": 1024,
            "bbox_coord_space": "abs",
            "bbox_format": "xyxy",
            "summary": {"caption": "x"},
            "items": [{"label": "object", "caption": "c", "palette": ["#fff"], "bbox": [0, 0, 8, 8]}],
        }
        parsed = json.loads(synth_layout.compile_draft(draft))
        assert list(parsed["items"][0]) == ["label", "caption", "palette", "bbox"]

    def test_rejects_unknown_fields(self):
        base = {
            "width": 1024,
            "height": 1024,
            "bbox_coord_space": "abs",
            "bbox_format": "xyxy",
            "summary": {"caption": "x"},
            "items": [{"label": "object", "caption": "c", "bbox": [0, 0, 8, 8]}],
        }
        bad = json.loads(json.dumps(base))
        bad["summary"]["children"] = []
        with pytest.raises(ValueError):
            synth_layout.compile_draft(bad)

        bad2 = json.loads(json.dumps(base))
        bad2["items"][0]["id"] = 1
        with pytest.raises(ValueError):
            synth_layout.compile_draft(bad2)

    def test_rejects_wrong_coord_space(self):
        draft = {
            "width": 1024,
            "height": 1024,
            "bbox_coord_space": "normalized_1000",
            "bbox_format": "xyxy",
            "summary": {"caption": "x"},
            "items": [{"label": "object", "caption": "c", "bbox": [0, 0, 8, 8]}],
        }
        with pytest.raises(ValueError):
            synth_layout.compile_draft(draft)


class TestSplitCanvas:
    def test_returns_requested_number_of_boxes(self):
        import random

        boxes = synth_layout.split_canvas(1024, 1024, 5, random.Random(0))
        assert len(boxes) == 5

    def test_boxes_inside_canvas_and_positive(self):
        import random

        for seed in range(20):
            boxes = synth_layout.split_canvas(1536, 1024, 6, random.Random(seed))
            for x1, y1, x2, y2 in boxes:
                assert 0 <= x1 < x2 <= 1536
                assert 0 <= y1 < y2 <= 1024

    def test_non_overlapping(self):
        import random

        boxes = synth_layout.split_canvas(1024, 1024, 4, random.Random(7))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                overlap_w = min(a[2], b[2]) - max(a[0], b[0])
                overlap_h = min(a[3], b[3]) - max(a[1], b[1])
                assert not (overlap_w > 0 and overlap_h > 0)


class TestGenerateLayouts:
    def test_deterministic_across_counts(self):
        first = [x["text"] for x in synth_layout.generate_layouts(50, seed=42)]
        longer = [x["text"] for x in synth_layout.generate_layouts(200, seed=42)]
        assert first == longer[:50]

    def test_different_seed_differs(self):
        a = [x["text"] for x in synth_layout.generate_layouts(20, seed=1)]
        b = [x["text"] for x in synth_layout.generate_layouts(20, seed=2)]
        assert a != b

    def test_output_is_valid_moss_layout(self):
        for record in synth_layout.generate_layouts(60, seed=7):
            parsed = json.loads(record["text"])
            assert set(parsed) == {
                "summary",
                "items",
                "bbox_coord_space",
                "bbox_format",
            }
            assert parsed["bbox_coord_space"] == "abs_centralized"
            assert parsed["bbox_format"] == "yxyx"
            assert isinstance(parsed["summary"], dict)
            assert parsed["items"]
            half_w, half_h = record["width"] // 2, record["height"] // 2
            for item in parsed["items"]:
                assert item["label"] in {
                    "object",
                    "character",
                    "text",
                    "panel",
                    "subject",
                }
                assert item["caption"]
                if item["label"] == "text":
                    assert item.get("text")
                y_min, x_min, y_max, x_max = item["bbox"]
                assert all(isinstance(v, int) for v in item["bbox"])
                assert y_min < y_max and x_min < x_max
                assert -half_h <= y_min and y_max <= half_h
                assert -half_w <= x_min and x_max <= half_w

    def test_variants_present(self):
        variants = {x["variant"] for x in synth_layout.generate_layouts(200, seed=3)}
        assert variants == {"comfy", "skill"}

    def test_language_mix(self):
        records = list(synth_layout.generate_layouts(400, seed=5, lang_mix=0.5))
        zh = sum(1 for r in records if r["lang"] == "zh")
        assert 140 <= zh <= 260
