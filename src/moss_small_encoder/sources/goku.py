"""Goku-OpenLab/gpt-image-2-prompts-datasets 读取器（中英双语提示词）。

约 1.96 万条，根目录一个 `metadata.jsonl`（115.8 MB），每条带 `raw_p`
和 `i18n.{en,zh}.{p,t,tags}`，中文侧是机器翻译。

已知问题，读的时候就要防：
- 中文译文质量不稳，偶尔出现「日文原句 + 中文替换」的混合句；
- 大量条目是 Midjourney 模板或「改这张图」的编辑指令，不是文生图描述。

为了不整包下载 115.8 MB，这里按字节区间等距抽样读取。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Record, jsonl_lines

REPO = "Goku-OpenLab/gpt-image-2-prompts-datasets"
REVISION = "5d347be762f9c8dc5bd0eaffb05778e47551855f"
METADATA = "metadata.jsonl"


def _safety_to_nsfw(rating: object) -> bool | None:
    """把 `spec.safety_rating` 转成三态。

    Args:
        rating: 原始评级字符串，例如 ``"Safe for Work"``。

    Returns:
        `"Safe for Work"` -> False；其他非空评级 -> True；缺失 -> None。
    """
    if not isinstance(rating, str) or not rating.strip():
        return None
    return rating.strip().lower() != "safe for work"


def iter_records(
    *, scan_rows: int | None = None, scan_bytes: int | None = None, cache_dir: Path | None = None
) -> Iterator[Record]:
    """读取含中文提示词的条目。

    Args:
        scan_rows: 本读取器不使用（JSONL 只能按字节抽）。
        scan_bytes: 最多抽样读取多少字节；None 表示整份顺序读完。
        cache_dir: 本读取器不使用。

    Yields:
        `text` 取自 `i18n.zh.p`，`nsfw` 取自 `spec.safety_rating`，
        `meta` 里带 category 与英文原文，便于人工抽查翻译质量。

    Notes:
        解析失败的行直接跳过：文件里有少量被截断或字段缺失的行。
    """
    del scan_rows, cache_dir
    for offset, line in jsonl_lines(REPO, REVISION, METADATA, scan_bytes=scan_bytes):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        i18n = row.get("i18n") or {}
        zh = (i18n.get("zh") or {}).get("p")
        if not isinstance(zh, str) or not zh.strip():
            continue
        spec = row.get("spec") or {}
        yield Record(
            text=zh,
            source="goku",
            source_id=str(row.get("id") or f"@{offset}"),
            nsfw=_safety_to_nsfw(spec.get("safety_rating")),
            meta={
                "category": row.get("category", ""),
                "en": (i18n.get("en") or {}).get("p", ""),
            },
        )
