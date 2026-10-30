"""Wikidata 实体抓取：拿「能在提示词里写出名字、且有具体形象」的实体清单喵。

类型不猜：`ROOT_TYPES` 里的 QID 全部来自锚点实测（蜜雪冰城/星巴克/麦当劳/
可口可乐/瑞幸/肯德基/耐克/苹果/米老鼠/皮卡丘/任天堂），
每个 QID 都至少被一个锚点的 P31 直接命中喵。

刻意**不用**这三个宽类型，实测会把没视觉形象的东西全带进来喵：
- Q43229 组织：连「联合国」都进来
- Q891723 上市公司：和「企业」重复，覆盖不增反降
- Q167270 商标：把「星球大战」这类作品名混成品牌

名字走 label 服务而不是 rdfs:label：`rdfs:label FILTER(LANG="zh")` 返回的是
繁体（蘋果公司、星際大戰），而 label 服务会按 `zh-hans,zh,en` 取第一个存在的喵。

输入: 无（联网走代理）
输出: jsonl，每行 {qid, name, sitelinks, type, root}
预期: 断点可续（按已抓 QID 去重）；按类型 + OFFSET 分页，单片失败重试喵
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "moss-image-mini-encoder/0.1 (dataset preparation)"
DEFAULT_PROXY = "http://127.0.0.1:7890"
PAGE = 5000
# 企业类实例量是品牌类的几十倍，带 ORDER BY 查询会直接超时（实测 HTTPError）喵。
# 单独给它更小的页，并且不排序——代价是分页顺序不保证稳定，靠 seen 去重兜底喵。
PAGE_OVERRIDE = {"Q4830453": 5000}
NO_ORDER = {"Q4830453"}
# 企业类里大量是没有中文/英文名的空壳条目，给个极低门槛压掉噪声喵。
MIN_SITELINKS = {"Q4830453": 1}

# 全部来自锚点实测（见 tmp/anchor_cover.py 的输出），不是凭印象写的喵。
ROOT_TYPES: dict[str, str] = {
    "Q431289": "品牌",
    "Q4830453": "企业",
    "Q386208": "吉祥物",
    "Q76212517": "连锁咖啡店",
    "Q18509232": "快餐连锁餐厅",
    "Q47504516": "饮品店",
    "Q76220181": "冰淇淋连锁店",
    "Q114392939": "饮料品牌",
}


def _to_simplified(name: str) -> str:
    """把繁体名转成简体喵。

    输入: name 实体名
    输出: 简体名；没装 zhconv 时原样返回
    预期: label 服务在 zh-hans 缺失时会回退到 zh（繁体），
          实测「三菱電機」「蘋果公司」就是这么来的，所以本地补一道喵
    """
    try:
        from zhconv import convert
    except ImportError:
        return name
    return convert(name, "zh-cn")


def _opener(proxy: str) -> urllib.request.OpenerDirector:
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def sparql(session: urllib.request.OpenerDirector, query: str, timeout: int = 130) -> list[dict]:
    """POST 一条 SPARQL，返回 bindings 列表喵。

    输入: session、query、timeout 秒
    输出: [{"item": "http://.../Q1", "name": "...", ...}]
    预期: 失败抛异常，由调用方重试
    """
    body = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
    )
    with session.open(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [{k: v.get("value") for k, v in row.items()}
            for row in payload["results"]["bindings"]]


def page_query(type_qid: str, offset: int, limit: int) -> str:
    """按类型取一页实例，名字用 label 服务（简体优先）喵。

    输入: type_qid 类型；offset 偏移；limit 页大小
    输出: SPARQL 字符串
    预期: 小类型带 ORDER BY 保证分页稳定；企业类不排序，避免超时
    """
    order = "" if type_qid in NO_ORDER else "ORDER BY ?item"
    floor = MIN_SITELINKS.get(type_qid, 0)
    floor_clause = f"FILTER(?n >= {floor})" if floor else ""
    return f"""
SELECT ?item ?name ?n WHERE {{
  ?item wdt:P31 wd:{type_qid} .
  ?item wikibase:sitelinks ?n .
  {floor_clause}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "zh-hans,zh,en".
    ?item rdfs:label ?name .
  }}
}}
{order}
LIMIT {limit} OFFSET {offset}
"""


QID_STEP = 5_000_000
LABEL_BATCH = 400
# 连续这么多片为空就认为扫到底了：QID 上限不写死，避免硬编码到 1.5 亿后漏掉尾部喵
MAX_EMPTY_SLICES = 3
ID_CACHE = Path("dataset/wikidata/enterprise_ids.jsonl")


def fetch_qids(
    session: urllib.request.OpenerDirector,
    type_qid: str,
    min_sitelinks: int,
    retries: int = 12,
    cache: Path | None = None,
    max_empty: int = MAX_EMPTY_SLICES,
) -> list[tuple[str, int]]:
    """按 QID 区间分片，只取 `(qid, sitelinks)`，不带名字喵。

    输入: session、type_qid、min_sitelinks 下限、retries、
          cache 缓存文件（存在就直接读，不重扫）、
          max_empty 连续多少片为空就停
    输出: [(qid, sitelinks), ...]
    预期: 不写死上限，扫到连续空片为止；失败重试不跳过喵

    Notes:
        两个设计点都是给「永远 20 分钟」这个问题修的喵：
        1. 上限不硬编码成 1.5 亿，改成连续空片判定；
        2. 结果落盘，补缺口时直接读缓存，不再重扫 30 片喵。

        为什么不带 label：`ORDER BY` + `OPTIONAL rdfs:label` 会把 WDQS 压到
        504（实测企业类必崩）。拆成两步后，这一步只做带索引的 P31 扫描喵。
    """
    if cache is not None and cache.is_file():
        cached: list[tuple[str, int]] = []
        with cache.open("r", encoding="utf-8") as fp:
            for line in fp:
                row = json.loads(line)
                cached.append((row["qid"], row["sitelinks"]))
        print(f"  读到 ID 缓存 {len(cached):,} 个 -> {cache}", flush=True)
        return cached

    floor = f"FILTER(?n >= {min_sitelinks})" if min_sitelinks else ""
    found: list[tuple[str, int]] = []
    empty_streak = 0
    lo = 0
    while empty_streak < max_empty:
        hi = lo + QID_STEP
        query = f"""
SELECT ?item ?n WHERE {{
  ?item wdt:P31 wd:{type_qid} .
  ?item wikibase:sitelinks ?n .
  {floor}
  BIND(xsd:integer(STRAFTER(STR(?item), "entity/Q")) AS ?qid)
  FILTER(?qid >= {lo} && ?qid < {hi})
}}
"""
        rows = None
        for attempt in range(1, retries + 1):
            try:
                rows = sparql(session, query, timeout=180)
                break
            except Exception as exc:
                wait = min(5 * attempt, 60)
                print(f"  [QID {lo:,}-{hi:,}] 第 {attempt} 次失败 "
                      f"{type(exc).__name__} {getattr(exc, 'code', '')}，等 {wait}s",
                      flush=True)
                time.sleep(wait)
        if rows is None:
            raise RuntimeError(
                f"{type_qid} 的 QID {lo}-{hi} 连续 {retries} 次失败；已拿到 {len(found)} 条"
            )
        for row in rows:
            found.append((row["item"].rsplit("/", 1)[-1], int(row["n"])))
        if rows:
            empty_streak = 0
            print(f"  [QID {lo:,}-{hi:,}] +{len(rows)} 累计 {len(found):,}", flush=True)
        else:
            empty_streak += 1
            print(f"  [QID {lo:,}-{hi:,}] 空片 {empty_streak}/{max_empty}", flush=True)
        lo += QID_STEP

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8") as fp:
            for qid, sits in found:
                fp.write(json.dumps({"qid": qid, "sitelinks": sits}) + "\n")
        print(f"  ID 缓存已落盘 {len(found):,} 个 -> {cache}", flush=True)
    return found


def fetch_labels(
    session: urllib.request.OpenerDirector,
    qids: list[str],
    retries: int = 12,
) -> dict[str, str]:
    """用 VALUES 批量取名字，中文优先、英文兜底，最后转简体喵。

    输入: session、qids
    输出: {qid: name}，取不到的被略过
    预期: 每批 LABEL_BATCH 个，失败重试不跳过
    """
    names: dict[str, str] = {}
    for start in range(0, len(qids), LABEL_BATCH):
        batch = qids[start:start + LABEL_BATCH]
        values = " ".join(f"wd:{qid}" for qid in batch)
        query = f"""
SELECT ?item ?zh ?en WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item rdfs:label ?zh FILTER(LANG(?zh) = "zh") }}
  OPTIONAL {{ ?item rdfs:label ?en FILTER(LANG(?en) = "en") }}
}}
"""
        rows = None
        for attempt in range(1, retries + 1):
            try:
                rows = sparql(session, query, timeout=180)
                break
            except Exception as exc:
                wait = min(5 * attempt, 60)
                print(f"  [label {start}-{start + len(batch)}] 第 {attempt} 次失败 "
                      f"{type(exc).__name__} {getattr(exc, 'code', '')}，等 {wait}s",
                      flush=True)
                time.sleep(wait)
        if rows is None:
            raise RuntimeError(f"label 批次 {start} 连续 {retries} 次失败")
        for row in rows:
            qid = row["item"].rsplit("/", 1)[-1]
            name = row.get("zh") or row.get("en") or ""
            name = _to_simplified(name.strip())
            if name and name != qid:
                names[qid] = name
        if (start // LABEL_BATCH) % 20 == 0:
            print(f"  label 进度 {start + len(batch):,}/{len(qids):,}", flush=True)
    return names
    """读出已抓到的 QID 集合，用于断点续抓喵。"""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                continue
    return done


def count_entities(
    session: urllib.request.OpenerDirector,
    type_qid: str,
    min_sitelinks: int = 0,
    retries: int = 6,
) -> int:
    """数某类型下带中/英文标签的唯一实体数喵。

    输入: session、type_qid、min_sitelinks 门槛、retries
    输出: 唯一实体数
    预期: 与抓取口径一致（同一 P31、同一 sitelinks 门槛、同样只算有名字的）

    Notes:
        **必须用 `COUNT(DISTINCT ?item)`**。一个实体同时有 zh 和 en 两个标签时，
        `COUNT(?item)` 会把它数两次——实测重复系数 1.210，
        把 106,909 数成了 126,918，直接导致我凭空造出两万条的缺口喵。
    """
    floor = f"FILTER(?n >= {min_sitelinks})" if min_sitelinks else ""
    query = f"""
SELECT (COUNT(DISTINCT ?item) AS ?c) WHERE {{
  ?item wdt:P31 wd:{type_qid} .
  ?item wikibase:sitelinks ?n .
  {floor}
  ?item rdfs:label ?x . FILTER(LANG(?x) IN ("zh", "en"))
}}
"""
    for attempt in range(1, retries + 1):
        try:
            rows = sparql(session, query, timeout=300)
            return int(rows[0]["c"])
        except Exception as exc:
            wait = min(10 * attempt, 60)
            print(f"  [count {type_qid}] 第 {attempt} 次失败 "
                  f"{type(exc).__name__} {getattr(exc, 'code', '')}，等 {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{type_qid} 的 COUNT(DISTINCT) 连续 {retries} 次失败")


def fetch_type(
    session: urllib.request.OpenerDirector,
    type_qid: str,
    seen: set[str],
    sink,
    retries: int = 3,
) -> int:
    """抓一个类型的全部实例，写入 sink喵。

    输入: session、type_qid、seen 已抓集合（就地更新）、sink 可写文件对象
    输出: 本类型新增条数
    预期: 某页返回空即停；每页失败重试 retries 次后放弃该页
    """
    added = 0
    offset = 0
    page_size = PAGE_OVERRIDE.get(type_qid, PAGE)
    while True:
        rows = None
        for attempt in range(1, retries + 1):
            try:
                rows = sparql(session, page_query(type_qid, offset, page_size))
                break
            except Exception as exc:
                detail = getattr(exc, "code", "")
                # 间隔递增：504 是服务端算太久，连着重试只会继续打它喵
                wait = min(5 * attempt, 60)
                print(f"  [{type_qid} offset={offset}] 第 {attempt} 次失败 "
                      f"{type(exc).__name__} {detail}，等 {wait}s", flush=True)
                time.sleep(wait)
        if rows is None:
            # 绝不静默跳过：跳过一页就等于永久丢掉那 5000 条，
            # 而且 OFFSET 分页跳过后无法事后定位漏了哪些喵
            raise RuntimeError(
                f"{type_qid} offset={offset} 连续 {retries} 次失败，"
                f"已抓 {added} 条；修复网络后重跑会按已抓 QID 续上喵"
            )
        if not rows:
            break
        for row in rows:
            qid = row["item"].rsplit("/", 1)[-1]
            name = _to_simplified(row.get("name", "").strip())
            # label 服务取不到中文/英文时会把 QID 自己当名字返回，这种没法用于提示词喵
            if not name or name == qid:
                continue
            if qid in seen:
                continue
            seen.add(qid)
            sink.write(json.dumps({
                "qid": qid,
                "name": name,
                "sitelinks": int(row["n"]),
                "type": type_qid,
                "root": ROOT_TYPES.get(type_qid, ""),
            }, ensure_ascii=False) + "\n")
            added += 1
        sink.flush()
        offset += page_size
        print(f"  {ROOT_TYPES.get(type_qid, type_qid)} offset={offset} "
              f"+{len(rows)} 累计新增 {added:,}", flush=True)
    return added


def fetch(out: Path, proxy: str) -> int:
    """抓全部类型，追加写入 jsonl喵。

    输入: out 输出路径；proxy
    输出: 落盘总条数
    预期: 同一 QID 只写一次，跨类型去重
    """
    session = _opener(proxy)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = already_done(out)
    print(f"已有 {len(seen):,} 条", flush=True)

    started = time.time()
    with out.open("a", encoding="utf-8") as fp:
        for type_qid in ROOT_TYPES:
            added = fetch_type(session, type_qid, seen, fp)
            print(f"[{ROOT_TYPES[type_qid]}] 新增 {added:,}，总计 {len(seen):,}，"
                  f"{time.time()-started:.0f}s", flush=True)
    return len(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓 Wikidata 品牌/吉祥物实体")
    parser.add_argument("--out", type=Path, default=Path("dataset/wikidata/entities.jsonl"))
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    args = parser.parse_args(argv)
    total = fetch(args.out, args.proxy)
    print(f"完成：{total:,} 条 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
