"""清洗规则的单测。每条规则都固定住「输入长什么样、输出该长什么样」。"""

import pytest

from moss_small_encoder import clean


class TestStripMjArguments:
    def test_replaces_default(self):
        text = '风格：一张 {argument name="character" default="Albedo"} Cosplay 快照'
        assert clean.strip_mj_arguments(text) == "风格：一张 Albedo Cosplay 快照"

    def test_replaces_typographic_quotes(self):
        # Goku 数据集用中文引号，和 shuixian 的 ASCII 引号不同。
        text = "{argument name=“style” default=“笨拙、潦草”}的方式重绘"
        assert clean.strip_mj_arguments(text) == "笨拙、潦草的方式重绘"

    def test_falls_back_to_name_without_default(self):
        text = "{argument name=\"theme\"} 为主题的游戏截图"
        assert clean.strip_mj_arguments(text) == "theme 为主题的游戏截图"

    def test_keeps_text_without_placeholder(self):
        text = "a red ball on a wooden table"
        assert clean.strip_mj_arguments(text) == text

    def test_handles_multiple_placeholders(self):
        text = "{argument name=\"a\" default=\"one\"} 和 {argument name=\"b\" default=\"two\"}"
        assert clean.strip_mj_arguments(text) == "one 和 two"


class TestFixGustavostaArtifacts:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ultra realistic, 8 k", "ultra realistic, 8k"),
            ("1 6 k resolution", "16k resolution"),
            ("sci - fi robotic factory", "sci-fi robotic factory"),
            ("3 d render", "3d render"),
            ("fine - face, realistic shaded", "fine-face, realistic shaded"),
        ],
    )
    def test_known_artifacts(self, raw, expected):
        assert clean.fix_gustavosta_artifacts(raw) == expected

    def test_keeps_normal_text(self):
        text = "a cat sitting on a mat, 4k, cinematic lighting"
        assert clean.fix_gustavosta_artifacts(text) == text

    def test_does_not_join_real_word_pairs(self):
        # 两个相邻单词不能被合并成 "ab"。
        text = "a b c"
        assert clean.fix_gustavosta_artifacts(text) == text


class TestStripDiffusionSyntax:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("masterpiece, <lora:add_detail:0.8>, 1girl", "masterpiece, , 1girl"),
            ("(masterpiece:1.2), best quality", "masterpiece, best quality"),
            ("((best quality)), ((detailed))", "best quality, detailed"),
            ("<lyco:foo:0.5> portrait", " portrait"),
        ],
    )
    def test_removes_syntax(self, raw, expected):
        assert clean.strip_diffusion_syntax(raw) == expected

    def test_keeps_plain_parens_content(self):
        assert clean.strip_diffusion_syntax("a (very) small cat") == "a very small cat"


class TestLooksLikeEditInstruction:
    @pytest.mark.parametrize(
        "text",
        [
            "在保持原图完全不变的前提下进行编辑，在画面周围添加迷你形象",
            "Redraw the attached image in the most clumsy way possible.",
            "Edit this image to add a hat",
            "保持画面完全不变，只改文字",
        ],
    )
    def test_detects_edit_prompts(self, text):
        assert clean.looks_like_edit_instruction(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "一个穿着 MOSI 标志 T 恤的中国美女面向镜头微笑",
            "A wide view of a refrigerator beside a kitchen counter with a fruit bowl.",
            "best quality, masterpiece, 1girl, blue hair",
        ],
    )
    def test_keeps_text_to_image_prompts(self, text):
        assert clean.looks_like_edit_instruction(text) is False


class TestNormalizeWhitespace:
    def test_unifies_newlines(self):
        assert clean.normalize_whitespace("a\r\nb\rc") == "a\nb\nc"

    def test_collapses_inner_spaces(self):
        assert clean.normalize_whitespace("a    b\t\tc") == "a b c"

    def test_strips_trailing_spaces_per_line(self):
        assert clean.normalize_whitespace("a   \nb  ") == "a\nb"

    def test_preserves_leading_whitespace(self):
        # 首部空白由调用方处理，这里不该悄悄改。
        assert clean.normalize_whitespace("  a") == " a"


class TestStripWrappingQuotes:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('"a red ball"', "a red ball"),
            ("“蓝色山脉插画”", "蓝色山脉插画"),
            ("「出师表」", "出师表"),
            ("no quotes here", "no quotes here"),
        ],
    )
    def test_strips_one_layer(self, raw, expected):
        assert clean.strip_wrapping_quotes(raw) == expected

    def test_unbalanced_quote_is_untouched(self):
        assert clean.strip_wrapping_quotes('"unbalanced') == '"unbalanced'


class TestDetectLang:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("一个穿着汉服的女性站在竹林里", "zh"),
            ("A woman in hanfu stands in a bamboo forest", "en"),
            ("a woman, 汉服, 竹林, soft light", "zh"),  # 中英混排归 zh
            ("", "en"),
        ],
    )
    def test_detect(self, text, expected):
        assert clean.detect_lang(text) == expected

    def test_cjk_ratio_empty(self):
        assert clean.cjk_ratio("   ") == 0.0


class TestDedupeKey:
    def test_identical_after_normalization(self):
        a = clean.dedupe_key("best quality,  1girl")
        b = clean.dedupe_key("BEST QUALITY, 1girl ")
        assert a == b

    def test_different_text_differs(self):
        assert clean.dedupe_key("abc") != clean.dedupe_key("abd")

    def test_does_not_dedupe_by_prefix(self):
        prefix = "best quality, masterpiece, "
        assert clean.dedupe_key(prefix + "cat") != clean.dedupe_key(prefix + "dog")


class TestSanitizeUnicode:
    def test_drops_lone_surrogate(self):
        assert clean.sanitize_unicode("a\ud800b") == "ab"

    def test_keeps_emoji(self):
        text = "emoji 🎬 and 中文"
        assert clean.sanitize_unicode(text) == text
