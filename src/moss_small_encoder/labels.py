"""NSFW 判定用的启发式关键词表。

只用于「分层抽样」和「分组报告指标」，不用于过滤掉成人内容——数据集要覆盖
NSFW 语料，否则投影在成人提示词上会退化。

判定刻意保守：命中明确词才算 Positive，宁可漏判。漏判只影响分桶统计的精度，
不会把样本弄丢或弄错。
"""

from __future__ import annotations

import re

# 明确的成人/裸露相关词，中英各一组。不收录仅属「性感」描述的词，
# 以免把普通写真误标成 NSFW。
_MARKERS_EN = re.compile(
    r"\b("
    r"nude|naked|topless|bottomless|undressed|exposed breasts|"
    r"explicit|nsfw|hentai|porn|pornographic|sex|sexual|erotic|"
    r"genitalia|genitals|penis|vagina|nipples|areola|"
    r"lingerie only|cameltoe|seductive pose nude"
    r")\b",
    re.IGNORECASE,
)

_MARKERS_ZH = re.compile(
    "裸露|赤裸|全裸|半裸|裸体|裸照|乳头|生殖器|色情|成人内容|性行为|情色"
)


def guess_nsfw(text: str) -> bool:
    """按关键词猜测一段提示词是否属于成人内容。

    Args:
        text: 提示词正文。

    Returns:
        命中任一明确标记词返回 True，否则 False。空文本返回 False。

    Notes:
        这是启发式，不是分类器。数据集里凡是源自己带标签的（例如 jtatman 的
        `nsfw_label`），一律优先用源标签，不调用本函数。
    """
    if not text:
        return False
    return bool(_MARKERS_EN.search(text) or _MARKERS_ZH.search(text))
