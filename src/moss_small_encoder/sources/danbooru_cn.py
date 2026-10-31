"""danbooru tag 中英对照数据源（ffdkj 表，MIT 许可）喵。

来源
----
https://github.com/ffdkj/ffdkj-Danbooru_Tag-Chinese-English-Translation-Table

每日更新，收录所有 `post_count >= 10` 的 tag，字段为
`name`（英文 tag）/ `category` / `cn_name`（中文译名）/ `post_count`喵。
译名由 Gemini 3 Flash 生成并经人工校对，许可 MIT喵。

为什么需要「共现」格式
----------------------
只给中文名时，教师的隐状态里不一定有对应外观；把中文名和英文 tag
写进同一个序列，两者在隐状态里共现，投影才学得到「这个中文写法 = 那个 tag」喵。

这也是不自己写属性词的原因：属性由教师提供，我们只负责把两种写法并列喵。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# 只取角色类 tag（danbooru 的 category=4）喵。
CHARACTER_CATEGORY = "4"

# 中性场景池：只描述环境，不含任何外观词喵。
SCENES = (
    "图书馆看书", "教室里坐着", "便利店里买饮料", "公园长椅上",
    "地铁车厢里", "天台上", "走廊里站着", "咖啡店里",
    "海边散步", "厨房做饭", "雨中撑伞", "雪地里",
    "森林里", "街道上", "卧室里", "阳台上",
    "车站里", "书店里", "超市里", "球场上",
)


def split_work(cn_name: str) -> tuple[str, str]:
    """把「鬼方佳代子（蔚蓝档案）」拆成名字与作品名喵。

    Args:
        cn_name: 对照表里的中文名，可能带全角括号出处。

    Returns:
        `(名字, 作品名)`；没有括号时作品名为空串。

    Notes:
        实测 9B 对不带作品名的名字会把名字画成画面上的文字，
        所以带出处的那部分必须保留，这里只是为了让句式更自然喵。
    """
    if "（" in cn_name and cn_name.endswith("）"):
        name, work = cn_name.rsplit("（", 1)
        return name.strip(), work[:-1].strip()
    return cn_name, ""


def make_text(cn_name: str, step: int, salt: int) -> str:
    """按序号产出「纯名」或「名字 + 场景」的文本喵。

    Args:
        cn_name: 中文名（可能带出处）。
        step: 该角色的第几个变体，0 为纯名。
        salt: 角色下标，用来打散场景，避免同一句被反复用。

    Returns:
        提示词文本。

    Notes:
        场景按下标错开抽样，同一角色的不同变体场景不同，
        不同角色也不会挤在同几个场景上喵。
    """
    if step == 0:
        return cn_name
    name, work = split_work(cn_name)
    scene = SCENES[(salt * 7 + step * 3) % len(SCENES)]
    if work:
        return f"{work}的{name}在{scene}"
    return f"{name}在{scene}"


@dataclass
class CnRecord:
    """一条待写入数据集的记录喵。

    Attributes:
        text: 提示词正文，中文名与英文 tag 可能并列。
        source_id: 对照表里英文 tag 名，用作业内唯一键。
        meta: 附带信息（英文 tag、中文名、post_count）。
        lang: 文本语种，纯中文名时是 `zh`，含英文 tag 时是 `mix`。
    """

    text: str
    source_id: str
    meta: dict = field(default_factory=dict)
    lang: str = "zh"


def load_pairs(
    csv_path: Path | str,
    min_post_count: int = 0,
    max_characters: int | None = None,
    seed: int = 20261027,
) -> list[dict]:
    """读对照表并筛出角色条目喵。

    Args:
        csv_path: `tags.csv` 路径。
        min_post_count: 只保留帖子数不低于该值的条目；0 表示不截断。
        max_characters: 最多取多少个角色，None 表示全部。
        seed: 洗牌种子；先按热度降序取前 N，再用该种子打乱喵。

    Returns:
        字典列表，键为 `name` / `cn_name` / `post_count`。

    Raises:
        FileNotFoundError: 文件不存在时给出下载命令提示喵。

    Notes:
        中文名或英文名缺失的条目直接跳过；`cn_name` 里可能带全角括号
        （如「鬼方佳代子（蔚蓝档案）」），这是译名的出处标注，必须保留喵。
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"找不到 {csv_path}；先跑 `python src/scripts/download_danbooru_cn.py`喵"
        )

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("category") != CHARACTER_CATEGORY:
                continue
            name = (row.get("name") or "").strip()
            cn_name = (row.get("cn_name") or "").strip()
            if not name or not cn_name:
                continue
            try:
                post_count = int(row.get("post_count") or 0)
            except ValueError:
                continue
            if post_count < min_post_count:
                continue
            rows.append({"name": name, "cn_name": cn_name, "post_count": post_count})

    rows.sort(key=lambda item: item["post_count"], reverse=True)
    if max_characters is not None:
        rows = rows[:max_characters]
    random.Random(seed).shuffle(rows)
    return rows


def iter_records(
    csv_path: Path | str,
    per_character: int = 4,
    seed: int = 20261027,
    min_post_count: int = 0,
    max_characters: int | None = None,
) -> Iterator[CnRecord]:
    """按模板产出中文名 + 英文 tag 的记录喵。

    Args:
        csv_path: 对照表路径。
        per_character: 每个角色产出几条，最多取 `len(TEMPLATES)`。
        seed: 洗牌与模板轮换种子，固定值保证可复现。
        min_post_count: 热度下限，0 表示不截断。
        max_characters: 最多取多少个角色，None 表示全部。

    Yields:
        `CnRecord`；`source_id` 形如 `kayoko_(blue_archive)#2`，带模板序号喵。

    Notes:
        模板按角色下标轮换，避免所有角色都用同一条模板，
        同时保证同一角色在不同运行里得到相同结果喵。
    """
    pairs = load_pairs(csv_path, min_post_count=min_post_count,
                       max_characters=max_characters, seed=seed)
    count = max(1, per_character)
    for index, item in enumerate(pairs):
        for step in range(count):
            yield CnRecord(
                text=make_text(item["cn_name"], step, index),
                source_id=f"{item['name']}#{step}",
                meta={
                    "en_tag": item["name"],
                    "cn_name": item["cn_name"],
                    "post_count": item["post_count"],
                },
                lang="zh",
            )


def digest(text: str) -> str:
    """算文本 sha1，供去重与复现校验喵。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


REPO = "ffdkj/ffdkj-Danbooru_Tag-Chinese-English-Translation-Table"
DEFAULT_CSV = Path("dataset/danbooru_cn/tags.csv")


def download(
    out: Path | str = DEFAULT_CSV,
    tag: str | None = None,
    proxy: str | None = "http://127.0.0.1:7890",
) -> Path:
    """从 GitHub release 拉取中英对照表喵。

    Args:
        out: 落盘路径，默认 `dataset/danbooru_cn/tags.csv`。
        tag: 钉死的 release 标签；None 表示取最新。
        proxy: HTTP 代理；None 表示直连喵。

    Returns:
        落盘后的路径喵。

    Notes:
        GitHub 的证书链在部分环境不完整，必须用 truststore 注入，
        否则报 `unable to get issuer certificate`喵。
        下载量约 14 MB，332,386 行喵。
    """
    import truststore

    truststore.inject_into_ssl()
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)

    suffix = f"/tags/{tag}" if tag else "/latest"
    api = f"https://api.github.com/repos/{REPO}/releases{suffix}"
    req = urllib.request.Request(api, headers={"User-Agent": "moss-encoder"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        release = json.loads(resp.read().decode("utf-8"))

    assets = {item["name"]: item for item in release.get("assets", [])}
    if "tags.csv" not in assets:
        raise RuntimeError(f"release {release.get('tag_name')} 里没有 tags.csv喵")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    asset = assets["tags.csv"]
    print(f"下载 tags.csv  {asset['size']:,} B -> {out}", flush=True)
    req = urllib.request.Request(asset["browser_download_url"],
                                 headers={"User-Agent": "moss-encoder"})
    written = 0
    with urllib.request.urlopen(req, timeout=300) as resp, out.open("wb") as fp:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fp.write(chunk)
            written += len(chunk)
    print(f"完成 {written:,} B  版本 {release.get('tag_name')}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Danbooru tag 中英对照表喵")
    parser.add_argument("--download", action="store_true", help="抓取对照表")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tag", default=None, help="钉死 release 标签")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--count", type=int, default=5, help="只打印前 N 条样例")
    parser.add_argument("--min-post-count", type=int, default=0)
    args = parser.parse_args(argv)

    if args.download:
        download(args.out, args.tag, args.proxy)
        return 0

    for record in iter_records(args.out, per_character=1,
                               min_post_count=args.min_post_count,
                               max_characters=args.count):
        print(f"{record.source_id:40} {record.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
