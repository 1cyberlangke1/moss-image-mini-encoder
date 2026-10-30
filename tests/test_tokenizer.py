"""分词器封装与 MOSS 模板的单测。

需要真实的 `tokenizer.json`；缺失时整组跳过，避免 CI 上没有权重就红。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moss_small_encoder import tokenizer as tok

TOKENIZER_PATH = Path("models/tokenizer/tokenizer.json")

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.is_file(), reason="缺少 models/tokenizer/tokenizer.json"
)


class TestWrap:
    def test_template_matches_generation_side(self):
        # 与 comfyui/moss-image/text_encoder.py 里的模板逐字一致。
        assert tok.wrap("abc") == (
            "<|im_start|>user\nabc<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )

    def test_strips_surrounding_whitespace(self):
        assert tok.wrap("  abc  ") == tok.wrap("abc")


class TestMossTokenizer:
    @pytest.fixture(scope="class")
    @staticmethod
    def moss():
        return tok.MossTokenizer.from_file(TOKENIZER_PATH)

    def test_wrapper_tokens_are_added_tokens(self, moss):
        ids = moss.encode("hello")
        text = moss.decode(ids)
        assert text.startswith("<|im_start|>user\nhello<|im_end|>")

    def test_count_matches_encode_length(self, moss):
        assert moss.count("a red ball") == len(moss.encode("a red ball"))

    def test_chinese_and_english_both_encode(self, moss):
        assert moss.count("一个穿汉服的女性") > 0
        assert moss.count("a woman in hanfu") > 0

    def test_long_text_exceeds_limit(self, moss):
        assert moss.count("word " * 6000) > tok.MAX_TEXT_TOKENS

    def test_empty_prompt_still_has_wrapper(self, moss):
        assert moss.count("") >= 10

    def test_layout_json_encodes(self, moss):
        text = (
            '{"summary":{"caption":"x"},"items":[{"label":"object","caption":"a cup",'
            '"bbox":[-288,-592,352,32]}],"bbox_coord_space":"abs_centralized",'
            '"bbox_format":"yxyx"}'
        )
        ids = moss.encode(text)
        assert len(ids) > 20
        assert "abs_centralized" in moss.decode(ids)
