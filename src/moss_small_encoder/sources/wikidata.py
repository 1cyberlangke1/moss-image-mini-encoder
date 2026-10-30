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
    预期: ORDER BY ?item 保证分页稳定，不会漏也不会重
    """
    return f"""
SELECT ?item ?name ?n WHERE {{
  ?item wdt:P31 wd:{type_qid} .
  ?item wikibase:sitelinks ?n .
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "zh-hans,zh,en".
    ?item rdfs:label ?name .
  }}
}}
ORDER BY ?item
LIMIT {limit} OFFSET {offset}
"""


def already_done(path: Path) -> set[str]:
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
    while True:
        rows = None
        for attempt in range(1, retries + 1):
            try:
                rows = sparql(session, page_query(type_qid, offset, PAGE))
                break
            except Exception as exc:
                print(f"  [{type_qid} offset={offset}] 第 {attempt} 次失败 "
                      f"{type(exc).__name__}", flush=True)
                time.sleep(3)
        if rows is None:
            print(f"  [{type_qid} offset={offset}] 放弃该页", flush=True)
            break
        if not rows:
            break
        for row in rows:
            qid = row["item"].rsplit("/", 1)[-1]
            name = row.get("name", "").strip()
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
        offset += PAGE
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
