"""BaYue-SYJ/shuixian-prompts 读取器（原生中文提示词）。

5,430 条，来自 Midjourney / Stable Diffusion / Flux / 即梦 / 可灵等平台的真实中文
提示词，含自然语言描述和 Midjourney 风格的结构化 JSON。

该仓库使用「Attribution Required License v1.0」：可以自由使用，但衍生项目必须在
显著位置署名并附原仓库链接。署名由 `dataset_card.md` 自动生成。

数据本身在 GitHub，不在 Hugging Face，所以按钉死的 commit sha 下载 raw 文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Record, http_text

REPO = "BaYue-SYJ/shuixian-prompts"
REVISION = "4b1500b6d15203f8c7b8853f2fff48e4758ab706"
PARTS = (
    "shuixian-unified/data/prompts.part1.json",
    "shuixian-unified/data/prompts.part2.json",
    "shuixian-unified/data/prompts.part3.json",
)


def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}"


def iter_records(
    *, scan_rows: int | None = None, scan_bytes: int | None = None, cache_dir: Path | None = None
) -> Iterator[Record]:
    """读取全部中文提示词。

    Args:
        scan_rows: 本读取器不使用。
        scan_bytes: 本读取器不使用（总共只有 10 MB，整份读完更简单）。
        cache_dir: 本地缓存根目录；首次运行会把三份 JSON 落盘复用。

    Yields:
        每条含 `prompt` 字段的条目转成一个 Record；
        `nsfw` 置 None（源用「大凶美女」「媚眼诱惑」这类分类名暗示成人向，
        但没有规范标签，交给启发式判定），分类名放在 `meta["category"]`。

    Notes:
        源里 58% 的条目带 Midjourney 变量占位符，清洗在 build 阶段做。
    """
    del scan_rows, scan_bytes
    if cache_dir is None:
        raise ValueError("shuixian 读取器需要 cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        local = cache_dir / Path(part).name
        if local.is_file():
            payload = json.loads(local.read_text(encoding="utf-8"))
        else:
            text = http_text(_raw_url(part))
            local.write_text(text, encoding="utf-8")
            payload = json.loads(text)
        for item in payload:
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            yield Record(
                text=prompt,
                source="shuixian",
                source_id=str(item.get("id", "")),
                nsfw=None,
                meta={
                    "category": item.get("category", ""),
                    "title": item.get("title", ""),
                },
            )
