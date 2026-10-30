"""MOSS-Image 仓库自带示例提示词读取器（保真度基准）。

`assets/images/*/parameters.json` 里存的是官方出图时真正用过的提示词。
这批样本只作为**测试集**使用，不参与训练：它的价值是「同一格式同分布下的
体检」，一旦进了训练集就失去参考意义。

三种 schema 并存，都保留并打标，统计时可以分开看：
- `canonical`：公开的 `summary` + `items` 格式；
- `scene_type`：官方内部草稿格式（`scene_type`/`gen_brief`/`elements`）；
- `plain`：纯文本提示词。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import Record, http_json, http_text

REPO = "OpenMOSS/MOSS-Image"
REVISION = "a211e398a33cbf4dd7a1cc52aba0dfdbac3b0281"


def _asset_paths() -> list[str]:
    """列出钉死 revision 下的示例参数文件。

    Args:
        无。

    Returns:
        `assets/images/*/parameters.json` 的相对路径列表（有序）。
    """
    tree = http_json(
        f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    )
    entries = tree.get("tree", []) if isinstance(tree, dict) else []
    return sorted(
        entry["path"]
        for entry in entries
        if entry.get("type") == "blob"
        and entry["path"].startswith("assets/images/")
        and entry["path"].endswith("parameters.json")
    )


def _classify(prompt: str) -> str:
    """判断示例提示词属于哪一种 schema。

    Args:
        prompt: 参数文件里的 prompt 字段。

    Returns:
        `"canonical"` / `"scene_type"` / `"plain"`。
    """
    try:
        parsed = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        return "plain"
    if not isinstance(parsed, dict):
        return "plain"
    if "summary" in parsed and isinstance(parsed.get("items"), list):
        return "canonical"
    if "scene_type" in parsed:
        return "scene_type"
    return "plain"


def iter_records(
    *, scan_rows: int | None = None, scan_bytes: int | None = None, cache_dir: Path | None = None
) -> Iterator[Record]:
    """读取官方示例提示词。

    Args:
        scan_rows: 本读取器不使用。
        scan_bytes: 本读取器不使用。
        cache_dir: 本地缓存根目录。

    Yields:
        `text` 为官方原始 prompt 字符串（不做任何清洗），
        `meta["schema"]` 标注来源 schema，`meta["id"]` 是目录名（001…020）。

    Notes:
        不做清洗是有意的：这是保真度基准，改了就不再是官方分布。
        只有 20 个文件，整份读完比抽样更简单，也不值得省这点流量。
    """
    del scan_rows, scan_bytes
    if cache_dir is None:
        raise ValueError("moss_official 读取器需要 cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"
    for path in _asset_paths():
        local = cache_dir / "moss_official" / Path(path).parent.name / "parameters.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file():
            payload = json.loads(local.read_text(encoding="utf-8"))
        else:
            text = http_text(base + path)
            local.write_text(text, encoding="utf-8")
            payload = json.loads(text)
        prompt = (payload.get("parameters") or {}).get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        yield Record(
            text=prompt,
            source="moss_official",
            source_id=Path(path).parent.name,
            kind="positive",
            meta={"schema": _classify(prompt), "path": path},
        )
