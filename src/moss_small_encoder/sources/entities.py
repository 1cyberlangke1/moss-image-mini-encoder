"""把 Wikidata 实体拼成提示词：只给名字，外观交给教师喵。

关键约束：**槽位里一个外观词都没有**喵。
不写颜色、不写服装、不写吉祥物长什么样，只写「名字 + 场景 + 视角 + 天气」。
这样教师看到的是一串陌生名字，它的隐状态里带着自己知道的外观，
我们学的是「名字 → 向量」，而不是「描述 → 向量」喵。
自己写描述会变成循环：等于先告诉它答案，再让它复述喵。

组合是纯枚举：8 场景 × 5 视角 × 6 天气 × 5 句式 = 1200 个变体/实体，
按实体名单独播种抽样，保证每个实体都被覆盖，且追加条数不影响别的实体喵。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterator

CHINESE_SCENES = ["店门口", "街角", "商场里", "柜台前", "夜市摊位", "广场上", "包装盒上", "广告牌上"]
CHINESE_VIEWS = ["特写", "远景", "仰拍", "平视", "俯视"]
CHINESE_TIMES = ["清晨", "正午", "黄昏", "雨夜", "雪天", "深夜"]

CHINESE_TEMPLATES = [
    "{x}，{scene}",
    "{x}，{scene}，{view}",
    "{x}，{scene}，{time}",
    "{x}，{view}，{time}",
    "给我画一张{x}的照片，{view}",
    "{x} 的招牌，{time}",
]

ENGLISH_SCENES = ["storefront", "street corner", "shopping mall", "counter", "night market", "plaza", "product packaging", "billboard"]
ENGLISH_VIEWS = ["close-up", "wide shot", "low angle", "eye level", "high angle"]
ENGLISH_TIMES = ["early morning", "noon", "dusk", "rainy night", "snowy day", "late night"]

ENGLISH_TEMPLATES = [
    "{x}, {scene}",
    "{x}, {scene}, {view}",
    "{x}, {scene}, {time}",
    "{x}, {view}, {time}",
    "a photo of {x}, {view}",
    "{x} logo, {time}",
]

# 名字里出现这些字符就当英文实体处理喵
_ASCII_ONLY = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '-&.")


def _is_ascii(name: str) -> bool:
    """判断实体名是不是纯 ASCII（决定用中文还是英文句式）喵。

    输入: name 实体名
    输出: True 表示纯 ASCII
    """
    return all(ch in _ASCII_ONLY for ch in name)


def make_variants(name: str, count: int, seed: int) -> list[str]:
    """给一个实体生成 count 条不重复的提示词喵。

    输入: name 实体名；count 条数；seed 全局种子（与实体名一起播种）
    输出: 提示词列表
    预期: 同一 (name, count, seed) 永远给出同一批；count 超过 1200 时封顶
    """
    if _is_ascii(name):
        scenes, views, times, templates = ENGLISH_SCENES, ENGLISH_VIEWS, ENGLISH_TIMES, ENGLISH_TEMPLATES
    else:
        scenes, views, times, templates = CHINESE_SCENES, CHINESE_VIEWS, CHINESE_TIMES, CHINESE_TEMPLATES

    combos: list[str] = []
    seen: set[str] = set()
    for template in templates:
        for scene in scenes:
            for view in views:
                for moment in times:
                    text = template.format(x=name, scene=scene, view=view, time=moment)
                    if text not in seen:
                        seen.add(text)
                        combos.append(text)

    rng = random.Random(f"{name}|{seed}")
    take = min(count, len(combos))
    return rng.sample(combos, take)


def load_entities(path: Path, min_sitelinks: int = 0) -> list[dict]:
    """读抓取结果，去重后按 sitelinks 降序喵。

    输入: path 抓取产物；min_sitelinks 门槛（0 表示全都留，含低频）
    输出: [{qid, name, sitelinks, root}]
    预期: 同名实体只保留 sitelinks 最高的那个
    """
    best: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if row["sitelinks"] < min_sitelinks:
                continue
            prev = best.get(row["name"])
            if prev is None or row["sitelinks"] > prev["sitelinks"]:
                best[row["name"]] = row
    return sorted(best.values(), key=lambda r: r["sitelinks"], reverse=True)


def iter_records(
    path: Path,
    per_entity: int,
    seed: int = 20261027,
    min_sitelinks: int = 0,
) -> Iterator[dict]:
    """把实体清单展开成提示词记录喵。

    输入: path 抓取产物；per_entity 每个实体生成多少条；seed；min_sitelinks
    输出: 记录 dict（字段与 prompts.jsonl 一致，缺 n_tokens 由 build 补）
    预期: id 唯一（含实体 qid 与序号），split 由 build 按 id 哈希决定
    """
    for entity in load_entities(path, min_sitelinks):
        variants = make_variants(entity["name"], per_entity, seed)
        for index, text in enumerate(variants):
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
            yield {
                "id": f"entity:{entity['qid']}:{index:03d}:{digest}",
                "bucket": "entity",
                "source": "wikidata",
                "source_id": entity["qid"],
                "kind": "positive",
                "lang": "en" if _is_ascii(entity["name"]) else "zh",
                "nsfw": False,
                "text": text,
                "meta": {"entity": entity["name"], "sitelinks": entity["sitelinks"]},
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wikidata 实体的提示词变体预览")
    parser.add_argument("--entities", type=Path, default=Path("dataset/wikidata/entities.jsonl"))
    parser.add_argument("--per-entity", type=int, default=8)
    parser.add_argument("--min-sitelinks", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20, help="预览多少条")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args(argv)

    entities = load_entities(args.entities, args.min_sitelinks)
    print(f"实体 {len(entities):,} 个（sitelinks>={args.min_sitelinks}）", flush=True)
    for index, record in enumerate(iter_records(args.entities, args.per_entity, args.seed,
                                               args.min_sitelinks)):
        if index >= args.limit:
            break
        print(f"  [{record['lang']}] {record['text']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
