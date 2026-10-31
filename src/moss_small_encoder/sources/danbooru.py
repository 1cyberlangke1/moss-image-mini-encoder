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
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

from .base import Record

API = "https://danbooru.donmai.us/tags.json"
PAGE_LIMIT = 1000
# page >= 1001 会返回 410 Gone；实测 page=1000 已经会 500，
# 所以单段 id 范围必须明显小于 100 万条，取 50 万喵
MAX_PAGES = 1000
ID_STEP = 500_000


def _max_id(session: urllib.request.OpenerDirector, retries: int = 8) -> int:
    """探测当前最大的 tag id，不写死上限喵。

    输入: session；retries 重试次数
    输出: 最大 id（探测失败抛错）
    预期: 用 `search[id]=>0` + `search[order]=id` 取降序第一条

    Notes:
        之前 `DEFAULT_RANGES` 写死到 300 万，只要 tag 继续增长，尾部就永远漏；
        这里改成每次运行先问服务端当前最大 id 是多少喵。
    """
    params = {"limit": 1, "search[id]": ">0", "search[order]": "id"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with session.open(req, timeout=60) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            if rows:
                return int(rows[0]["id"])
        except OSError as exc:
            wait = min(5 * attempt, 60)
            print(f"  [探测最大 id] 第 {attempt} 次失败 {type(exc).__name__}，等 {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError("探测 Danbooru 最大 tag id 失败")


def id_ranges(top: int, step: int = ID_STEP) -> list[tuple[int, int]]:
    """按最大 id 切出覆盖全库的区间列表喵。

    输入: top 最大 id；step 单段宽度
    输出: [(lo, hi), ...]，末段右端覆盖 top
    预期: 段数随 top 增长自动增加，不存在写死的上限喵
    """
    return [(lo, min(lo + step, top + 1)) for lo in range(0, top + 1, step)]
USER_AGENT = "moss-image-mini-encoder/0.1"

# 角色名后面的槽位：**一个外观词都没有**喵。
# 写「黑发红瞳」这类就是在替教师回答，学到的会变成「描述 → 向量」，
# 而我们真正要的是「名字 → 向量」，外观必须由教师自己提供喵。
CHARACTER_TEMPLATES = [
    "{x}",
    "{x}，特写",
    "{x}，远景",
    "{x}，站姿，全身",
    "{x}，侧面，半身",
    "画一张{x}",
    "{x}，图书馆里看书",
    "{x}，教室里坐着",
    "{x}，便利店里买饮料",
    "{x}，公园长椅上",
    "{x}，地铁车厢里",
    "{x}，天台上",
    "{x}，走廊里站着",
    "{x}，咖啡店里",
    "{x}，海边散步",
    "{x}，厨房做饭",
    "{x}，雨中撑伞",
    "{x}，雪地里",
    "{x}，森林里",
    "{x}，街道上",
    "{x}，卧室里",
    "{x}，阳台上",
    "{x}，车站里",
    "{x}，书店里",
    "{x}，超市里",
    "{x}，球场上",
]

# Danbooru 的 category 编号：0 通用、1 画师、3 作品、4 角色、5 meta
CATEGORY_GENERAL = 0
CATEGORY_ARTIST = 1
CATEGORY_SERIES = 3
CATEGORY_CHARACTER = 4
CATEGORY_META = 5


def _opener(proxy: str) -> urllib.request.OpenerDirector:
    """按需装代理，返回可复用的 opener喵。"""
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _fetch_page(session: urllib.request.OpenerDirector, page: int,
                lo: int = 0, hi: int = 0) -> list[dict]:
    """取第 page 页；返回空列表表示翻到底了喵。

    输入: session；page 页码；lo/hi 给了就按 id 区间过滤
    输出: tag 记录列表
    预期: lo=hi=0 表示不限制区间（旧的整表分页行为）
    """
    params = {"limit": PAGE_LIMIT, "page": page}
    if hi:
        params["search[id]"] = f"{lo}..{hi}"
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with session.open(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _last_page(session: urllib.request.OpenerDirector, lo: int, hi: int,
               retries: int = 8) -> int:
    """二分出某个 id 区间的最后一个非空页，0 表示区间为空喵。

    输入: session；lo/hi id 区间；retries 单页重试次数
    输出: 页号
    预期: 约 log2(1000) ≈ 10 次请求定位末页，避免逐页翻到空喵

    Notes:
        这正是之前空转 666 页的修法：区间里有多少条是**可以提前量出来**的，
        量出来之后只翻这么多页，一页都不多请求喵。
    """
    def probe(page: int) -> list[dict]:
        for attempt in range(1, retries + 1):
            try:
                return _fetch_page(session, page, lo, hi)
            except OSError as exc:
                wait = min(5 * attempt, 60)
                print(f"  [探测 {lo}-{hi} 第 {page} 页] 第 {attempt} 次失败 "
                      f"{type(exc).__name__}，等 {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"探测 {lo}-{hi} 第 {page} 页连续 {retries} 次失败")

    if not probe(1):
        return 0
    valid, invalid = 1, MAX_PAGES + 1
    while valid + 1 < invalid:
        mid = (valid + invalid) // 2
        if probe(mid):
            valid = mid
        else:
            invalid = mid
    return valid


def download_tags(out: Path, proxy: str = "http://127.0.0.1:7890",
                  sleep: float = 0.25, report_every: int = 20,
                  ranges: list[tuple[int, int]] | None = None,
                  retries: int = 8) -> int:
    """抓全库 tag 清单，一行一条 jsonl，断了能续喵。

    输入: out 落地路径；proxy 为空则直连；sleep 每页间隔秒数；
          ranges id 区间列表，None 用默认三段；retries 单页重试次数
    输出: 落盘总行数
    预期: 每个区间只翻它实际的页数；失败重试不跳过；同一 id 只写一次喵

    Notes:
        两条硬限制都必须在设计里体现，不能靠跑的时候发现喵：
        1. `page >= 1001` 返回 410 Gone，单条查询最多 100 万条；
        2. 区间条数可提前二分量出，翻够就停，否则会空转几百页喵。
    """
    session = _opener(proxy)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ranges is None:
        top = _max_id(session)
        ranges = id_ranges(top)
        print(f"当前最大 tag id {top:,}，切成 {len(ranges)} 段", flush=True)

    seen: set[int] = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    seen.add(int(json.loads(line)["id"]))
                except Exception:
                    continue
        print(f"续抓：已有 {len(seen)} 条", flush=True)

    written = len(seen)
    mode = "a" if seen else "w"
    with out.open(mode, encoding="utf-8") as fp:
        for lo, hi in ranges:
            pages = _last_page(session, lo, hi, retries=retries)
            if pages == 0:
                print(f"[{lo}-{hi}] 区间为空，跳过", flush=True)
                continue
            print(f"[{lo}-{hi}] 共 {pages} 页，开始抓", flush=True)
            for page in range(1, pages + 1):
                rows = None
                for attempt in range(1, retries + 1):
                    try:
                        rows = _fetch_page(session, page, lo, hi)
                        break
                    except OSError as exc:
                        wait = min(5 * attempt, 60)
                        print(f"[{lo}-{hi}] 第 {page} 页第 {attempt} 次失败 "
                              f"{type(exc).__name__}，等 {wait}s", flush=True)
                        time.sleep(wait)
                if rows is None:
                    # 绝不跳过：跳过一页等于丢掉那 1000 条，事后无法定位喵
                    raise RuntimeError(
                        f"[{lo}-{hi}] 第 {page} 页连续 {retries} 次失败，"
                        f"已写 {written} 条；重跑会按已有 id 续上喵"
                    )
                fresh = 0
                for row in rows:
                    if row["id"] in seen:
                        continue
                    seen.add(row["id"])
                    fp.write(json.dumps({
                        "id": row["id"],
                        "name": row["name"],
                        "post_count": row["post_count"],
                        "category": row["category"],
                        "is_deprecated": row["is_deprecated"],
                    }, ensure_ascii=False) + "\n")
                    fresh += 1
                fp.flush()
                written += fresh
                if report_every and page % report_every == 0:
                    print(f"[{lo}-{hi}] 第 {page} 页，累计 {written} 条", flush=True)
                time.sleep(sleep)
            print(f"[{lo}-{hi}] 区间完成，累计 {written} 条", flush=True)
    return written


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


def iter_characters(
    tags_path: Path,
    per_character: int,
    seed: int = 20261027,
    max_characters: int | None = None,
    min_post_count: int = 0,
    templates: list[str] | None = None,
) -> Iterator[Record]:
    """按角色展开变体，**只给名字、不给任何属性词**。

    输入: tags_path 全库 tag 清单；per_character 每个角色多少条；
          max_characters 最多取多少角色（None 表示全要）；
          min_post_count 热度下限（0 表示低频也要）
    输出: `Record` 流，source 为 `danbooru`
    预期: 每个角色都被覆盖；同一角色的所有变体共享 source_id

    Notes:
        为什么不像 `iter_records` 那样配属性：通用属性是随机抽的，
        给黑发角色配上 `blonde_hair` 会把错误绑定喂进教师喵。
        外观必须由教师自己提供，我们只搬运「名字 → 向量」喵。

        名字给两种写法：danbooru 原生 `kayoko_(blue_archive)` 和可读的
        `kayoko (blue archive)`，因为用户两种都可能写喵。
    """
    tags = load_tags(tags_path)
    characters = [r for r in tags if r["category"] == CATEGORY_CHARACTER]
    if min_post_count:
        characters = [r for r in characters if r["post_count"] >= min_post_count]
    if max_characters is not None:
        characters = characters[:max_characters]
    if not characters:
        raise ValueError("tag 清单里没有角色，检查下载是否完整")

    slots = templates or CHARACTER_TEMPLATES
    for character in characters:
        raw = character["name"]
        readable = _pretty(raw)
        variants: list[str] = []
        for template in slots:
            variants.append(template.format(x=readable))
            if raw != readable:
                variants.append(template.format(x=raw))
        picked = _sample_strings(variants, per_character, f"{raw}|{seed}")
        for index, text in enumerate(picked):
            yield Record(
                text=text,
                source="danbooru",
                source_id=f"{character['id']}:{index:03d}",
                lang="en",
                nsfw=None,
                kind="positive",
                meta={"character": raw, "post_count": character["post_count"]},
            )


ARTIST_TEMPLATES = (
    "best quality, highres, 1girl, solo, @{a}",
    "best quality, highres, 1boy, solo, @{a}",
    "best quality, highres, scenery, no humans, @{a}",
    "best quality, highres, 1girl, upper body, @{a}",
)

QUALITY_TAGS = ("best quality", "highres")


def iter_artists(
    tags_path: Path,
    per_artist: int,
    seed: int = 20261027,
    max_artists: int | None = None,
    min_post_count: int = 0,
    templates: tuple[str, ...] | None = None,
) -> Iterator[Record]:
    """按画师展开变体，只给画师名，不给任何属性词喵。

    输入: tags_path 全库 tag 清单；per_artist 每个画师多少条；
          max_artists 上限；min_post_count 热度下限
    输出: `Record` 流，source 为 `danbooru`
    预期: 每个画师都被覆盖；同一画师的变体共享 source_id 前缀

    Notes:
        画师 tag 在 danbooru 里是裸名，官方示例写成 `@foo bar`，
        所以这里统一加 `@` 前缀，让教师看到的是「画师引用」而不是普通物件喵。

        主体固定在 1girl / 1boy / scenery 三种，
        不随机抽通用属性：随机属性会把画师的风格信号冲掉喵。
    """
    tags = load_tags(tags_path)
    artists = [r for r in tags if r["category"] == CATEGORY_ARTIST]
    if min_post_count:
        artists = [r for r in artists if r["post_count"] >= min_post_count]
    artists.sort(key=lambda r: (-r["post_count"], r["name"]))
    if max_artists is not None:
        artists = artists[:max_artists]
    if not artists:
        raise ValueError("tag 清单里没有画师，检查下载是否完整")

    slots = templates or ARTIST_TEMPLATES
    for artist in artists:
        raw = artist["name"]
        readable = _pretty(raw)
        variants = [t.format(a=readable) for t in slots]
        if raw != readable:
            variants += [t.format(a=raw) for t in slots]
        picked = _sample_strings(variants, per_artist, f"{raw}|artist|{seed}")
        for index, text in enumerate(picked):
            yield Record(
                text=text,
                source="danbooru",
                source_id=f"{artist['id']}:a{index:03d}",
                lang="en",
                nsfw=None,
                kind="positive",
                meta={"artist": raw, "post_count": artist["post_count"]},
            )


# meta 类（category 5）几乎全是标注动作与画质标记，不是画面内容，
# 官方推荐顺序里的 Meta 一档因此在 danbooru 词表上无对应，整档跳过喵。

def _usable_general(name: str) -> bool:
    """过滤掉纯标点、纯数字这类没法当画面描述的 general tag喵。

    输入: tag 名
    输出: 能不能用
    预期: 必须含至少一个 ASCII 字母，且首字符必须是字母喵
    """
    if not name or not name[0].isalpha():
        return False
    return any(ch.isalpha() and ch.isascii() for ch in name)


def iter_tag_sequences(
    tags_path: Path,
    count: int,
    seed: int = 20261027,
    min_post_count: int = 50,
) -> Iterator[Record]:
    """按官方推荐顺序拼真实 tag 序列喵。

    输入: tags_path；count 条数；min_post_count 各分组热度下限
    输出: `Record` 流
    预期: 顺序恒为 quality -> year -> artist -> general

    Notes:
        官方例子是 `best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl,
        blue hair`，顺序照它，写错了模型认不出来喵。

        官方顺序里还有 Score 与 Meta 两档，在 danbooru 词表上没有对应：
        `score_9` 属于别的词表体系，meta 类几乎全是 `tagme` / `bad_id` /
        `costume_request` 这种标注动作。硬塞进来等于教模型两个不存在的字段，
        所以这两档整档跳过喵。

        **这里不放 Character / Series**：general tag 是随机抽的，
        给 `kayoko_(blue_archive)` 配上 `blonde_hair` 会把错误绑定喂进教师，
        和角色桶「只给名字、属性由教师提供」的原则直接冲突喵。
        角色的名字绑走 `characters` 与 `characters_cn` 两个桶喵。

        其余每个 tag 都取自 danbooru 真实词表，一个都不是我们编的喵。
        `min_post_count` 挡掉 0 帖死 tag（全库 82% 的 general 是 0 帖）喵。
    """
    tags = load_tags(tags_path)
    pool = [r for r in tags if r["post_count"] >= min_post_count]
    general = [
        r["name"]
        for r in pool
        if r["category"] == CATEGORY_GENERAL and _usable_general(r["name"])
    ]
    # 年份只收 2010 以后的：danbooru 里 1945 这类数字 tag 是别的东西，不是创作年份喵
    years = [
        r["name"]
        for r in tags
        if re.fullmatch(r"(201\d|202\d)", r["name"]) and r["post_count"] >= 1000
    ]
    artists = [r["name"] for r in pool if r["category"] == CATEGORY_ARTIST]
    if not general:
        raise ValueError("tag 清单里没有足够通用 tag")

    rng = random.Random(seed)
    for index in range(count):
        parts = list(QUALITY_TAGS)
        if years and rng.random() < 0.4:
            parts.append(rng.choice(years))
        if artists and rng.random() < 0.6:
            parts.append("@" + _pretty(rng.choice(artists)))
        parts.extend(rng.sample(general, min(rng.randint(5, 9), len(general))))
        yield Record(
            text=", ".join(parts),
            source="danbooru",
            source_id=f"tagseq:{index:06d}",
            lang="en",
            nsfw=None,
            kind="positive",
            meta={"n_tags": len(parts)},
        )


def _sample_strings(pool: list[str], count: int, salt: str) -> list[str]:
    """从候选里确定性抽 count 条，不足则全给喵。

    输入: pool 候选；count 目标条数；salt 播种用盐
    输出: 去重后的列表
    预期: 同一 (pool, count, salt) 永远给出同一批
    """
    uniq: list[str] = []
    seen: set[str] = set()
    for text in pool:
        if text not in seen:
            seen.add(text)
            uniq.append(text)
    if count >= len(uniq):
        return uniq
    return random.Random(salt).sample(uniq, count)


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
