"""跨数据源共用的文本清洗规则。

全部是纯函数：输入 str，输出 str 或 bool，不做 IO、不读配置、不依赖随机数。
这样清洗规则可以被单测固定住，改动前后能直接对比条数影响。

规则按「这个数据源有什么已知污染」设计，不追求通用清洗。
"""

from __future__ import annotations

import hashlib
import re

# Midjourney 的变量占位符，shuixian 和 Goku 两个源里大量出现。
# 形态：{argument name="style" default="clumsy"} 或 {argument name=“theme” default=“撮り鉄”}
# 有 default 就替换成 default 的文本；没有 default 时退化成变量名，保留一点语义。
_MJ_ARGUMENT = re.compile(
    r"""\{argument\s+name\s*=\s*["“](?P<name>[^"”]*)["”]"""
    r"""(?:\s+default\s*=\s*["“](?P<default>[^"”]*)["”])?\s*\}""",
    re.X,
)


def strip_mj_arguments(text: str) -> str:
    """把 Midjourney 变量占位符替换成它的默认值。

    Args:
        text: 可能含 `{argument name="x" default="y"}` 的原始提示词。

    Returns:
        占位符被替换后的文本。有 default 用 default，没有则用变量名。
        不含占位符时原样返回（不做任何其他修改）。
    """

    def replace(match: re.Match[str]) -> str:
        default = match.group("default")
        return default if default is not None else match.group("name")

    return _MJ_ARGUMENT.sub(replace, text)


# Gustavosta 的数据在发布前被错误地按空格切过一次：
# "8 k"、"1 6 k"、"sci - fi"。真实分词里这些应该连在一起，
# 不修会让模型学到一个现实中不存在的分词习惯。
_SPLIT_DIGITS = re.compile(r"\b(\d) (\d)\b")
# 单位前不能要求词边界：`16 k` 里 `6` 前面是数字，`\b` 会拦住它。
# 用「数字+空格+单位字母且后面不接字母」来锁定，`8 k` 和 `16 k` 都能命中。
_SPLIT_UNIT = re.compile(r"(\d) ([kKmMgGdD])(?![A-Za-z])")
_SPACED_HYPHEN = re.compile(r"(\w) - (\w)")


def fix_gustavosta_artifacts(text: str) -> str:
    """修复 Gustavosta 数据集里「数字/单位/连字符被空格拆开」的伪影。

    Args:
        text: 原始提示词，可能含 `8 k`、`1 6 k`、`sci - fi`。

    Returns:
        修复后的文本：`8k`、`16k`、`sci-fi`。其他内容不变。

    Notes:
        三条规则要连续应用两轮：`1 6 k` 第一轮变 `16 k`，第二轮才变 `16k`。
    """
    for _ in range(2):
        text = _SPLIT_DIGITS.sub(r"\1\2", text)
        text = _SPLIT_UNIT.sub(r"\1\2", text)
    return _SPACED_HYPHEN.sub(r"\1-\2", text)


# Civitai 提示词里的权重语法，对 MOSS 是无意义字符。
_STRONG_TAGS = re.compile(r"<(?:lora|lyco|hypernet|ti|embedding|spiece):[^>]*>", re.I)
_WEIGHTED_PAREN = re.compile(r"\(([^():]*):\s*[0-9]*\.?[0-9]+\s*\)")
_PLAIN_PAREN = re.compile(r"\(([^()]*)\)")


def strip_diffusion_syntax(text: str) -> str:
    """移除 Stable Diffusion 的权重语法，只留词语本身。

    Args:
        text: 含 `<lora:name:0.8>`、`(masterpiece:1.2)`、`((best quality))` 的提示词。

    Returns:
        去掉 LoRA/嵌入标记、拆掉括号与权重数字后的文本。
        `(masterpiece:1.2)` -> `masterpiece`；`((best quality))` -> `best quality`。
    """
    text = _STRONG_TAGS.sub("", text)
    text = _WEIGHTED_PAREN.sub(r"\1", text)
    # 只拆一层，重复两次以覆盖 ((x)) 这种常见写法。
    for _ in range(2):
        text = _PLAIN_PAREN.sub(r"\1", text)
    return text


# 「改图指令」不是文生图提示词：它要求保持一张已有图片不变。
# 这类样本混进来，投影会学到「编辑」而不是「从文字生成」的映射。
_EDIT_MARKERS_ZH = (
    "保持原图",
    "在原图",
    "原图不变",
    "这张图的基础上",
    "把这张图",
    "这张照片",
    "保持画面完全不变",
    "进行编辑",
)
_EDIT_MARKERS_EN = (
    "redraw the attached",
    "attached image",
    "edit this image",
    "edit the image",
    "the provided image",
    "the original image",
    "image-to-image",
    "keep the original image",
)


def looks_like_edit_instruction(text: str) -> bool:
    """判断一段提示词是不是「编辑已有图片」的指令，而不是文生图描述。

    Args:
        text: 待判定的提示词。

    Returns:
        True 表示命中编辑指令特征，应当从文生图训练集里剔除。

    Notes:
        只做关键词命中，宁可漏判也不误杀：误杀会损失正常样本，
        漏判只是混进少量噪声。所以标记词都取不会出现在正常描述里的短语。
    """
    lowered = text.lower()
    return any(marker in text for marker in _EDIT_MARKERS_ZH) or any(
        marker in lowered for marker in _EDIT_MARKERS_EN
    )


_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.M)


def normalize_whitespace(text: str) -> str:
    """统一换行符并压缩多余空格，但保留换行结构。

    Args:
        text: 任意文本，可能有 CRLF、行尾空格、连续多个空格。

    Returns:
        CRLF 统一成 LF；每行行尾空格去掉；行内两个以上空格压成一个。
        换行本身保留，因为 MOSS 用换行控制文字渲染的断行。

    Notes:
        不做 strip()：首尾空白由调用方决定是否处理，避免悄悄改变语义。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    return _TRAILING_SPACE.sub("", text)


_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}


def strip_wrapping_quotes(text: str) -> str:
    """去掉整体包住文本的一层引号。

    Args:
        text: 可能被 `"..."` 或 `“...”` 整个包住的提示词。

    Returns:
        去掉一层首尾配对引号并 strip 之后的文本；引号不成对时只 strip 首尾空白，
        不吃掉那个落单的引号。
    """
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in _QUOTE_PAIRS:
        if stripped[-1] == _QUOTE_PAIRS[stripped[0]]:
            return stripped[1:-1].strip()
    return stripped


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def cjk_ratio(text: str) -> float:
    """计算文本里汉字占非空白字符的比例。

    Args:
        text: 任意文本。

    Returns:
        0.0~1.0 的比例。空文本或全空白返回 0.0。
    """
    dense = re.sub(r"\s", "", text)
    if not dense:
        return 0.0
    return len(_CJK.findall(dense)) / len(dense)


def detect_lang(text: str) -> str:
    """按汉字比例判断语种，用于分桶与分层划分。

    Args:
        text: 任意文本。

    Returns:
        `"zh"` 当汉字占非空白字符 15% 以上，否则 `"en"`。

    Notes:
        阈值取 15% 而不是 50%：中文用户常写「英文标签 + 中文描述」的混排提示词，
        `a woman, 汉服, 竹林, soft light` 这种汉字只占 18% 的句子归到 zh
        更符合实际使用。这个标签只用于分桶与分层抽样，不影响文本本身。
    """
    return "zh" if cjk_ratio(text) >= 0.15 else "en"


_WS_RUN = re.compile(r"\s+")


def dedupe_key(text: str) -> str:
    """生成用于去重的稳定键。

    Args:
        text: 提示词原文。

    Returns:
        小写、空白折叠后全文的 sha1 十六进制串。

    Notes:
        用全文而不是前缀：两个不同的长提示词可能共享同样的开头
        （比如都以 "best quality, masterpiece, " 起手），按前缀去重会误杀。
    """
    normalized = _WS_RUN.sub(" ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def text_sha1(text: str) -> str:
    """计算原文（不做规范化）的 sha1，用于 manifest 里追踪具体样本。

    Args:
        text: 提示词原文。

    Returns:
        sha1 十六进制串。
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sanitize_unicode(text: str) -> str:
    """清洗非法 Unicode 代理项，保证能安全写成 UTF-8 JSONL。

    Args:
        text: 从 parquet/jsonl 里读出的原始字符串。

    Returns:
        去掉孤立代理项后的文本；正常文本原样返回。
    """
    return text.encode("utf-8", errors="ignore").decode("utf-8")
