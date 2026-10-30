"""jtatman/stable-diffusion-prompts-stats-full-uncensored 读取器。

896,874 行、9 列，来自 Civitai 的 Stable Diffusion 提示词，带 `nsfw_label`
与 `nsfw_score` 两个标注列，所以成人内容判定不需要靠猜。

只读需要的四列，并只读均匀散布的若干行组：取 1.2 万条不需要传 200 MB。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Record, parquet_groups

REPO = "jtatman/stable-diffusion-prompts-stats-full-uncensored"
REVISION = "eeddd630cca52dbb9af5a4a40aa5748b565da36e"

COLUMNS = ["prompt", "negative_prompt", "nsfw_label", "image_id"]


def _nsfw(value: object) -> bool | None:
    """把源标签转成三态：True / False / None。

    Args:
        value: `nsfw_label` 列的原始值。

    Returns:
        `"NSFW"` -> True，`"SFW"` -> False，其他（空或未知）-> None。
    """
    if not isinstance(value, str):
        return None
    label = value.strip().upper()
    if label == "NSFW":
        return True
    if label == "SFW":
        return False
    return None


def iter_records(
    *, scan_rows: int | None = None, scan_bytes: int | None = None, cache_dir: Path | None = None
) -> Iterator[Record]:
    """流式读取正负提示词。

    Args:
        scan_rows: 最多读多少行；None 表示读全部 89 万行。
        scan_bytes: 本读取器不使用。
        cache_dir: 本读取器不使用（数据直接从远程按需读）。

    Yields:
        `kind="positive"` 与 `kind="negative"` 两种记录。负面提示词单独成条，
        因为生成时它走同一个文本编码器。

    Notes:
        `source_id` 里带文件名、行组内偏移和原 image_id，便于回溯到具体一行。
    """
    del scan_bytes, cache_dir
    for name, offset, table in parquet_groups(
        REPO, REVISION, "data/*.parquet", COLUMNS, scan_rows=scan_rows
    ):
        prompts = table.column("prompt").to_pylist()
        negatives = table.column("negative_prompt").to_pylist()
        labels = table.column("nsfw_label").to_pylist()
        ids = table.column("image_id").to_pylist()
        stem = Path(name).stem
        for i, (prompt, negative) in enumerate(zip(prompts, negatives)):
            row_id = f"{stem}:{offset + i}:{ids[i]}"
            nsfw = _nsfw(labels[i])
            if isinstance(prompt, str):
                yield Record(
                    text=prompt,
                    source="jtatman",
                    source_id=row_id,
                    nsfw=nsfw,
                    kind="positive",
                )
            if isinstance(negative, str) and negative.strip():
                yield Record(
                    text=negative,
                    source="jtatman",
                    source_id=row_id,
                    nsfw=nsfw,
                    kind="negative",
                )
