"""Gustavosta/Stable-Diffusion-Prompts 读取器。

81,910 行（train 73,718 / test 8,192），单列 `Prompt`，是 SD 1.x 时代的
artstation 风格提示词。数据在发布前被按空格切过一次（`8 k` / `sci - fi`），
清洗时用 `clean.fix_gustavosta_artifacts` 修复。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Record, parquet_groups

REPO = "Gustavosta/Stable-Diffusion-Prompts"
REVISION = "d816d4a05cb89bde39dd99284c459801e1e7e69a"


def iter_records(
    *, scan_rows: int | None = None, scan_bytes: int | None = None, cache_dir: Path | None = None
) -> Iterator[Record]:
    """流式读取提示词。

    Args:
        scan_rows: 最多读多少行；None 表示读全部训练集。
        scan_bytes: 本读取器不使用。
        cache_dir: 本读取器不使用（数据直接从远程按需读）。

    Yields:
        原始字符串记录，`nsfw` 为 None（该数据集没有标注，由启发式判定）。
    """
    del scan_bytes, cache_dir
    for name, offset, table in parquet_groups(
        REPO, REVISION, "data/train*.parquet", ["Prompt"], scan_rows=scan_rows
    ):
        stem = Path(name).stem
        for i, prompt in enumerate(table.column("Prompt").to_pylist()):
            if not isinstance(prompt, str):
                continue
            yield Record(
                text=prompt,
                source="gustavosta",
                source_id=f"{stem}:{offset + i}",
            )
