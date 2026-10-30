"""Danbooru tag 源：抓全库 tag 清单，再拼成提示词记录。

两条路分开：
1. `download_tags()` 走 Danbooru 公开接口分页，产出 tag 清单（含画师）；
2. `iter_records()` 从清单里抽「角色 + 外观属性」的组合，拼成三种格式的提示词。

只给 tag 名时，教师模型看到的是一串陌生字符串，不带外观信息；
所以拼的时候必须让角色 tag 和属性 tag 同时出现在同一条里，
才能把「名字 → 属性」的共现喂进去喵。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

from .base import Record

API = "https://danbooru.donmai.us/tags.json"
PAGE_LIMIT = 1000
USER_AGENT = "moss-image-mini-encoder/0.1"

# Danbooru 的 category 编号：0 通用、1 画师、3 作品、4 角色、5 meta
CATEGORY_GENERAL = 0
CATEGORY_SERIES = 3
CATEGORY_CHARACTER = 4


def _opener(proxy: str) -> urllib.request.OpenerDirector:
    """按需装代理，返回可复用的 opener喵。"""
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _fetch_page(session: urllib.request.OpenerDirector, page: int) -> list[dict]:
    """取第 page 页；返回空列表表示翻到底了喵。"""
    query = urllib.parse.urlencode({"limit": PAGE_LIMIT, "page": page})
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with session.open(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_tags(out: Path, proxy: str = "http://127.0.0.1:7890",
                  sleep: float = 0.25, report_every: int = 20) -> int:
    """抓全库 tag 清单，一行一条 jsonl，断了能续喵。

    输入: out 落地路径；proxy 为空则直连；sleep 每页间隔秒数
    输出: 落盘总行数
    预期: 抓到空页为止；文件已存在时按行数换算页码继续，不重头抓
    """
    session = _opener(proxy)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out.exists():
        with out.open("r", encoding="utf-8") as fp:
            done = sum(1 for _ in fp)
        print(f"续抓：已有 {done} 条", flush=True)

    page = done // PAGE_LIMIT + 1
    mode = "a" if done else "w"
    with out.open(mode, encoding="utf-8") as fp:
        while True:
            try:
                rows = _fetch_page(session, page)
            except OSError as exc:
                # 抓一次要十几分钟，网络的临时抖动不该让进度白费喵
                print(f"第 {page} 页失败 {type(exc).__name__}，5 秒后重试", flush=True)
                time.sleep(5)
                continue
            if not rows:
                break
            for row in rows:
                fp.write(json.dumps({
                    "id": row["id"],
                    "name": row["name"],
                    "post_count": row["post_count"],
                    "category": row["category"],
                    "is_deprecated": row["is_deprecated"],
                }, ensure_ascii=False) + "\n")
            fp.flush()
            done += len(rows)
            if report_every and page % report_every == 0:
                print(f"第 {page} 页，累计 {done} 条", flush=True)
            page += 1
            time.sleep(sleep)
    return done


def load_tags(path: Path) -> list[dict]:
    """把 jsonl 读成列表，顺手剔掉废弃 tag喵。

    输入: download_tags 产出的文件
    输出: 记录列表（字段同文件），按 post_count 降序
    """
    tags: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if row["is_deprecated"]:
                continue
            tags.append(row)
    tags.sort(key=lambda r: r["post_count"], reverse=True)
    return tags


def _pretty(name: str) -> str:
    """把 danbooru 的 snake_case 名字还原成能读的样子喵。

    输入: `hatsune_miku`
    输出: `hatsune miku`
    """
    return name.replace("_", " ")


def load_attr_pool(tags_dir: Path) -> list[str]:
    """读「外观属性」子表，避免从全量通用 tag 里抽到物件和梗喵。

    输入: 含 feature.csv / attire.csv / expression.csv 的目录
    输出: tag 名列表（去重）
    预期: 目录不存在或文件缺失时返回空列表，由调用方退回全量通用 tag
    """
    names: list[str] = []
    for fname in ("feature.csv", "attire.csv", "expression.csv"):
        path = tags_dir / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            next(reader, None)  # 跳过表头 tag,count
            for row in reader:
                # 带括号的是「某作品某角色 cosplay」这类专名，不是外观属性喵
                if row and "(" not in row[0] and "cosplay" not in row[0]:
                    names.append(row[0])
    seen: set[str] = set()
    uniq: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _sample_attrs(rng: random.Random, pool: list[str], k: int) -> list[str]:
    """从属性词表里抽 k 个喵。

    输入: rng；pool 为可读 tag 名列表；k 为个数
    输出: k 个可读 tag 名（下划线换成空格）
    预期: k 超过词表长度时按词表长度返回
    """
    picked = rng.sample(pool, min(k, len(pool)))
    return [_pretty(n) for n in picked]


def iter_records(tags_path: Path, count: int, seed: int = 20261027,
                 top_characters: int = 20000,
                 attr_dir: Path | None = Path("dataset/danbooru_tags")) -> Iterator[Record]:
    """按「角色 + 属性」拼提示词，三种格式轮换喵。

    输入: tags_path 全库 tag 清单；count 目标条数；seed 随机种子；
          attr_dir 属性词表目录（有就只从外观属性里抽）
    输出: `Record` 流，source 为 `danbooru`
    预期: 角色取热度前 top_characters 个；每条 4~10 个属性；
          格式按 index%3 轮换：纯 tag 串 / 逗号串 / 自然句
    """
    tags = load_tags(tags_path)
    characters = [r for r in tags if r["category"] == CATEGORY_CHARACTER][:top_characters]
    series = {r["name"] for r in tags if r["category"] == CATEGORY_SERIES}
    attr_pool = load_attr_pool(attr_dir) if attr_dir else []
    if not attr_pool:
        # 没有分类词表就退回全量通用 tag，抽到的会杂一些喵
        attr_pool = [r["name"] for r in tags if r["category"] == CATEGORY_GENERAL][:5000]
    if not characters or not attr_pool:
        raise ValueError("tag 清单里没有角色或属性 tag，检查下载是否完整")

    rng = random.Random(seed)
    for i in range(count):
        char = rng.choice(characters)
        # 角色 tag 尾巴常带作品名，这里把它当作品标注用喵
        base = _pretty(char["name"])
        attrs = _sample_attrs(rng, attr_pool, rng.randint(4, 10))
        has_series = any(part in series for part in char["name"].split("_"))

        if i % 3 == 0:
            text = ", ".join([char["name"], *[a.replace(" ", "_") for a in attrs]])
        elif i % 3 == 1:
            text = ", ".join([base, *attrs])
        else:
            text = f"a picture of {base}, " + ", ".join(attrs)

        yield Record(
            text=text,
            source="danbooru",
            source_id=str(char["id"]),
            lang="en",
            nsfw=None,
            kind="positive",
            meta={"character": char["name"], "series_tag": has_series},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Danbooru tag 清单与提示词生成")
    parser.add_argument("--tags", type=Path, default=Path("dataset/danbooru_api/tags.jsonl"))
    parser.add_argument("--download", action="store_true", help="先抓全库 tag 清单")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--count", type=int, default=10, help="只做预览时生成的条数")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args(argv)

    if args.download:
        total = download_tags(args.tags, proxy=args.proxy)
        print(f"tag 清单 {total} 条 -> {args.tags}", flush=True)

    for rec in iter_records(args.tags, args.count, args.seed):
        print(rec.text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
