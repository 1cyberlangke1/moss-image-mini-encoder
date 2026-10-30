"""n714/stable-diffusion-prompt-library-zh_CN 读取器（中文标签词表）。

仓库里是 `tags/*.yml`，结构为「中文标签: 英文标签」的嵌套字典。
它的价值是提供**中文标签流**：MOSS 支持 tag 序列输入，而中文用户写标签流时
用的是「黑发, 长发, 汉服」这种中文词，不是英文。

这里只提供词表和确定性组合，不直接产出提示词。
"""

from __future__ import annotations

import random
import urllib.parse
from pathlib import Path
from typing import Iterable, Iterator

import yaml

from .base import http_json, http_text

REPO = "n714/stable-diffusion-prompt-library-zh_CN"
REVISION = "31199613c8ed198eb6fa74b20cc63725d568e62f"

# 中文标签至少要有这么多个汉字才算「中文词条」，过滤掉 "A-1" 这种纯编号键。
_MIN_CJK = 1


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _walk_leaves(node: object, out: list[str]) -> None:
    """递归收集叶子键名（中文侧）。

    Args:
        node: yaml 解析出的子树。
        out: 收集结果，原地追加。

    Notes:
        只收「值是非空字符串」的键：这样的键才是 `中文: english` 的标签对，
        中间层的分类名（如「头发颜色」）不是可以直接拼进提示词的词。
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and value.strip() and _has_cjk(str(key)):
                if sum("\u4e00" <= c <= "\u9fff" for c in str(key)) >= _MIN_CJK:
                    out.append(str(key))
            else:
                _walk_leaves(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk_leaves(item, out)


def tag_files() -> list[str]:
    """列出钉死 revision 下的标签文件。

    Args:
        无。

    Returns:
        仓库内 `tags/*.yml` 的相对路径列表（有序）。
    """
    tree = http_json(
        f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    )
    entries = tree.get("tree", []) if isinstance(tree, dict) else []
    return sorted(
        entry["path"]
        for entry in entries
        if entry.get("type") == "blob"
        and entry["path"].startswith("tags/")
        and entry["path"].endswith(".yml")
    )


def chinese_tags(cache_dir: Path) -> list[str]:
    """下载并汇总全部中文标签，去重后按字典序返回。

    Args:
        cache_dir: 本地缓存根目录。

    Returns:
        去重后的中文标签列表。

    Notes:
        结果排序后返回，保证同一 revision 下顺序稳定，
        否则组合抽样会随文件系统顺序漂移。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    collected: list[str] = []
    for path in tag_files():
        local = cache_dir / "n714" / Path(path).name
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file():
            text = local.read_text(encoding="utf-8")
        else:
            url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{urllib.parse.quote(path)}"
            text = http_text(url)
            local.write_text(text, encoding="utf-8")
        _walk_leaves(yaml.safe_load(text), collected)
    return sorted(set(collected))


def compose(
    tags: Iterable[str], count: int, seed: int, min_len: int = 4, max_len: int = 12
) -> Iterator[str]:
    """把中文标签组合成 tag 流式的提示词。

    Args:
        tags: 候选中文标签（会被转成有序列）。
        count: 生成条数。
        seed: 随机种子，决定每条样本的随机源。
        min_len: 每条至少包含几个标签。
        max_len: 每条最多包含几个标签。

    Yields:
        形如 ``"黑发, 长发, 汉服, 柔和侧光"`` 的字符串。

    Notes:
        标签不重复，用 `random.sample` 保证；每条独立播种，
        所以先跑少量与跑满结果一致。
    """
    pool = list(dict.fromkeys(tags))
    if not pool:
        raise ValueError("标签池为空，无法组合")
    for index in range(count):
        rng = random.Random(seed * 100_003 + index)
        k = rng.randint(min_len, min(max_len, len(pool)))
        yield ", ".join(rng.sample(pool, k))
