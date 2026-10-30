"""构建训练集：读公开源 → 清洗 → 数 token → 去重 → 分层划分 → 落盘与报告。

设计原则：

- **可复现**：每个源钉死 commit sha；随机器用固定种子；同一份配置重跑结果逐字节一致。
- **不猜**：超过 4096 token 的样本整条剔除（不是截断），并单独计数，避免长提示词的
  覆盖缺口被悄悄抹平。
- **只读外部**：外部仓库只作为数据来源被读取，代码不 import 它们。

输出（默认在 `dataset/`）：

    prompts.jsonl      一行一条训练样本（不入库）
    manifest.json      源 revision、条数、剔除原因、包版本、产物 sha256
    stats.md           各桶构成、token 分布、NSFW 与语种比例
    dataset_card.md    来源、许可、署名、已知限制
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.metadata as md
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import yaml
from tqdm import tqdm

from . import clean, labels, synth_layout
from .sources import SOURCES, goku, gustavosta, jtatman, moss_official, n714, shuixian
from .tokenizer import MAX_TEXT_TOKENS, MossTokenizer

# 源 -> 清洗函数。每个函数只做「这个词源已知的污染」。
CLEANERS: dict[str, Callable[[str], str]] = {
    "jtatman": lambda t: clean.normalize_whitespace(
        clean.strip_wrapping_quotes(clean.strip_diffusion_syntax(t))
    ).strip(),
    "gustavosta": lambda t: clean.normalize_whitespace(
        clean.strip_wrapping_quotes(
            clean.strip_diffusion_syntax(clean.fix_gustavosta_artifacts(t))
        )
    ).strip(),
    "shuixian": lambda t: clean.normalize_whitespace(
        clean.strip_wrapping_quotes(clean.strip_mj_arguments(t))
    ).strip(),
    "goku": lambda t: clean.normalize_whitespace(
        clean.strip_wrapping_quotes(clean.strip_mj_arguments(t))
    ).strip(),
    "n714": lambda t: clean.normalize_whitespace(t).strip(),
    "moss_official": lambda t: t,
    "synth_layout": lambda t: t,
}

# 数据源许可，写进 dataset_card.md。上游没标许可的如实写「未标注」。
LICENSES = {
    "jtatman": "未标注（源自 Civitai，使用前请自行确认）",
    "gustavosta": "未标注（数据集卡片 license 为 unknown）",
    "shuixian": "Attribution Required License v1.0（需署名）",
    "goku": "CC BY 4.0",
    "n714": "MIT",
    "moss_official": "Apache-2.0（随 MOSS-Image 仓库）",
    "synth_layout": "本仓库自生成",
}

SOURCE_URLS = {
    "jtatman": "https://huggingface.co/datasets/jtatman/stable-diffusion-prompts-stats-full-uncensored",
    "gustavosta": "https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts",
    "shuixian": "https://github.com/BaYue-SYJ/shuixian-prompts",
    "goku": "https://huggingface.co/datasets/Goku-OpenLab/gpt-image-2-prompts-datasets",
    "n714": "https://github.com/n714/stable-diffusion-prompt-library-zh_CN",
    "moss_official": "https://github.com/OpenMOSS/MOSS-Image",
    "synth_layout": "https://github.com/OpenMOSS/MOSS-Image",
}

SOURCE_REVISIONS = {
    "jtatman": jtatman.REVISION,
    "gustavosta": gustavosta.REVISION,
    "shuixian": shuixian.REVISION,
    "goku": goku.REVISION,
    "n714": n714.REVISION,
    "moss_official": moss_official.REVISION,
    "synth_layout": "n/a",
}


def load_config(path: Path) -> dict:
    """读 YAML 配置。

    Args:
        path: 配置文件路径。

    Returns:
        配置字典。
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def drop_reason(source: str, kind: str, text: str, cfg: dict) -> str | None:
    """按源与桶判断一条清洗后的文本是否该丢弃。

    Args:
        source: 源短名。
        kind: `"positive"` 或 `"negative"`。
        text: 已清洗的文本。
        cfg: 全局配置（用到 `min_chars` / `max_chars`）。

    Returns:
        丢弃原因的短标签；保留时返回 None。

    Notes:
        长度上限用字符数做粗筛，真正的 4096 token 上限在数完 token 之后再判。
    """
    if len(text) < cfg["min_chars"]:
        return "too_short"
    if len(text) > cfg["max_chars"]:
        return "too_long_chars"
    if source == "shuixian" and clean.looks_like_edit_instruction(text):
        return "edit_instruction"
    if source == "goku":
        if clean.looks_like_edit_instruction(text):
            return "edit_instruction"
        if clean.detect_lang(text) != "zh":
            return "not_zh"
    return None


def _priority(item: dict) -> int:
    """给候选打一个稳定的伪随机优先级。

    Args:
        item: 含 `source` 与 `source_id` 的候选记录。

    Returns:
        64 位整数优先级，越小越优先保留。

    Notes:
        用 `source|source_id` 的哈希而不是内容哈希：内容哈希会让两条一模一样
        但来自不同源的样本拿到同一优先级，而它们本来就在去重阶段被处理掉了。
    """
    raw = f"{item.get('source', '')}|{item.get('source_id', '')}".encode("utf-8")
    return int(hashlib.sha1(raw).hexdigest()[:16], 16)


def reservoir(
    items: Iterator[dict], quota: int, seed: int, stats: Counter
) -> list[dict]:
    """从候选里取出优先级最高的 `quota` 条。

    Args:
        items: 逐条产出的记录（已经过滤掉该丢的）。
        quota: 目标条数。
        seed: 保留参数以统一调用签名，当前实现不依赖随机种子。
        stats: 计数器，记录候选总数与超额丢弃数。

    Returns:
        最多 `quota` 条记录，按优先级从高到低排列。

    Notes:
        这里不是经典蓄水池抽样，而是「按稳定哈希取前 N」：
        配额从 500 提到 3000 时，前 500 条原样保留、只往后追加，
        这样同一份数据可以做嵌套的数据量缩放实验。
    """
    del seed  # 抽样只依赖 source_id 的哈希，保证配额变化时结果可嵌套
    heap: list[tuple[int, int, dict]] = []
    counter = 0
    for item in items:
        stats["candidates"] += 1
        priority = _priority(item)
        if len(heap) < quota:
            heapq.heappush(heap, (-priority, counter, item))
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, (-priority, counter, item))
            stats["over_quota"] += 1
        else:
            stats["over_quota"] += 1
        counter += 1
    return [item for _, _, item in sorted(heap, key=lambda entry: -entry[0])]


def split_of(record_id: str, ratios: dict[str, float]) -> str:
    """按记录 id 的哈希稳定地分配 train/val/test。

    Args:
        record_id: 全局唯一 id。
        ratios: 形如 `{"train": 0.7, "val": 0.15, "test": 0.15}`。

    Returns:
        划分名。

    Notes:
        用哈希而不是随机数：同一份数据在任何机器、任何顺序下都分到同一侧，
        且与内容无关，因此不会把相似样本整批塞进同一侧。
    """
    bucket = int(hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    train = ratios["train"]
    val = train + ratios["val"]
    if bucket < train:
        return "train"
    if bucket < val:
        return "val"
    return "test"


def _iter_stream(bucket: dict, cfg: dict) -> Iterator[dict]:
    """按配置从某个流式源里产出候选记录。

    Args:
        bucket: 桶配置（含 source / record / quota）。
        cfg: 全局配置。

    Yields:
        含 `text` `source_id` `nsfw` `kind` `meta` 的字典，文本已清洗，
        但尚未做长度与 token 判定。
    """
    source = bucket["source"]
    module = SOURCES[source]
    want = bucket.get("record", "positive")
    cleaner = CLEANERS[source]
    for record in module.iter_records(
        scan_rows=bucket.get("scan_rows"),
        scan_bytes=bucket.get("scan_bytes"),
        cache_dir=Path(cfg["cache_dir"]),
    ):
        if record.kind != want:
            continue
        text = cleaner(clean.sanitize_unicode(record.text))
        yield {
            "text": text,
            "source": source,
            "source_id": record.source_id,
            "kind": want,
            "nsfw": record.nsfw,
            "meta": record.meta,
        }


def _iter_synth(bucket: dict, cfg: dict) -> Iterator[dict]:
    """产出合成的 layout 提示词。

    Args:
        bucket: 桶配置。
        cfg: 全局配置（用到 seed）。

    Yields:
        文本已就绪、无需清洗的记录。
    """
    for item in synth_layout.generate_layouts(
        bucket["quota"], cfg["seed"], bucket.get("lang_mix", 0.5)
    ):
        yield {
            "text": item["text"],
            "source": "synth_layout",
            "source_id": f"{item['variant']}:{item['width']}x{item['height']}:{item['n_items']}",
            "kind": "positive",
            "nsfw": False,
            "meta": {
                "variant": item["variant"],
                "width": item["width"],
                "height": item["height"],
                "n_items": item["n_items"],
            },
            "lang": item["lang"],
        }


def _iter_tags(bucket: dict, cfg: dict) -> Iterator[dict]:
    """产出由中文标签组合成的 tag 流提示词。

    Args:
        bucket: 桶配置。
        cfg: 全局配置。

    Yields:
        文本已就绪的记录。
    """
    tags = n714.chinese_tags(Path(cfg["cache_dir"]))
    for index, text in enumerate(
        n714.compose(tags, bucket["quota"], cfg["seed"] + 7)
    ):
        yield {
            "text": clean.normalize_whitespace(text).strip(),
            "source": "n714",
            "source_id": f"compose:{index}",
            "kind": "positive",
            "nsfw": False,
            "meta": {"tags": len(text.split(", "))},
        }


def _iter_official(bucket: dict, cfg: dict) -> Iterator[dict]:
    """产出 MOSS-Image 官方示例提示词（只做测试集）。

    Args:
        bucket: 桶配置。
        cfg: 全局配置。

    Yields:
        文本保持官方原样、未经清洗的记录。
    """
    for record in moss_official.iter_records(cache_dir=Path(cfg["cache_dir"])):
        yield {
            "text": record.text,
            "source": "moss_official",
            "source_id": record.source_id,
            "kind": "positive",
            "nsfw": False,
            "meta": record.meta,
        }


ITERATORS: dict[str, Callable[[dict, dict], Iterator[dict]]] = {
    "stream": _iter_stream,
    "synth": _iter_synth,
    "tag_compose": _iter_tags,
    "official": _iter_official,
}


def collect_bucket(
    bucket: dict,
    cfg: dict,
    tokenizer: MossTokenizer,
    seen: set[str],
    forced_split: str | None,
    max_read: int | None = None,
) -> tuple[list[dict], Counter]:
    """收集一个桶的全部样本。

    Args:
        bucket: 桶配置。
        cfg: 全局配置。
        tokenizer: 用于数 token。
        seen: 全局已用去重键集合，跨桶共享。
        forced_split: 非 None 时该桶全部样本强制进这个划分（官方样例用）。
        max_read: 只读这么多条候选就停，用于冒烟测试；None 表示读全量。

    Returns:
        `(记录列表, 计数器)`。记录已含 `id` `split` `n_tokens` `lang` `nsfw`。

    Notes:
        计数器的键就是 `stats.md` 里「为什么丢」那一列，不要随意改名。
    """
    stats: Counter = Counter()
    source = bucket.get("source", "synth_layout")
    iterator = ITERATORS[bucket["kind"]](bucket, cfg)
    quota = bucket["quota"]

    def candidates() -> Iterator[dict]:
        for item in iterator:
            if max_read is not None and stats["read"] >= max_read:
                return
            stats["read"] += 1
            text = item["text"]
            if not text:
                stats["drop:empty"] += 1
                continue
            reason = drop_reason(item.get("source", source), item.get("kind", "positive"), text, cfg)
            if reason is not None:
                stats[f"drop:{reason}"] += 1
                continue
            n_tokens = tokenizer.count(text)
            if n_tokens > cfg["max_tokens"]:
                stats["drop:over_max_tokens"] += 1
                continue
            if n_tokens == 0:
                stats["drop:zero_tokens"] += 1
                continue
            key = clean.dedupe_key(text)
            if key in seen:
                stats["drop:duplicate"] += 1
                continue
            seen.add(key)
            item["n_tokens"] = n_tokens
            yield item

    kept = reservoir(candidates(), quota, cfg["seed"], stats)

    rows = []
    for item in kept:
        record_id = f"{bucket['name']}:{item['source']}:{item['source_id']}"
        text = item["text"]
        lang = item.get("lang") or clean.detect_lang(text)
        nsfw = item["nsfw"]
        if nsfw is None:
            nsfw = labels.guess_nsfw(text)
        rows.append(
            {
                "id": record_id,
                "bucket": bucket["name"],
                "source": item["source"],
                "source_id": item["source_id"],
                "kind": item["kind"],
                "lang": lang,
                "nsfw": bool(nsfw),
                "n_tokens": item["n_tokens"],
                "split": forced_split or split_of(record_id, cfg["splits"]),
                "text_sha1": clean.text_sha1(text),
                "text": text,
            }
        )
    stats["kept"] = len(rows)
    return rows, stats


def _package_versions() -> dict[str, str]:
    """记录构建时用到的包版本，便于别人对照环境。

    Args:
        无。

    Returns:
        包名到版本的映射；缺失的包跳过。
    """
    versions = {"python": sys.version.split()[0]}
    for name in ("huggingface-hub", "tokenizers", "pyarrow", "numpy", "pyyaml"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return versions


def _token_summary(values: list[int]) -> dict[str, float]:
    """给一组 token 数算统计量。

    Args:
        values: token 数列表。

    Returns:
        均值、中位数、p95、最大值；空列表全部返回 0。
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 2),
        "median": round(float(np.median(array)), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "max": float(array.max()),
    }


def write_stats(
    path: Path,
    rows: list[dict],
    stats_by_bucket: dict[str, Counter],
    order: list[str],
    cfg: dict,
) -> None:
    """生成 `stats.md`。

    Args:
        path: 输出路径。
        rows: 全部保留样本。
        stats_by_bucket: 每个桶的计数器。
        order: 桶的输出顺序。
        cfg: 全局配置。

    Returns:
        无。直接写文件。
    """
    lines = ["# 数据集统计", ""]
    lines.append(f"- 随机种子：`{cfg['seed']}`")
    lines.append(f"- token 上限：`{cfg['max_tokens']}`（超过则整条剔除，不截断）")
    lines.append(f"- 样本总数：**{len(rows)}**")
    total_tokens = sum(r["n_tokens"] for r in rows)
    lines.append(f"- token 总数：**{total_tokens:,}**")
    lines.append("")

    lines.append("## 各桶构成")
    lines.append("")
    lines.append("| 桶 | 读到 | 保留 | 剔除原因 | 中位 token | p95 token | NSFW | 中文占比 |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for name in order:
        bucket_rows = [r for r in rows if r["bucket"] == name]
        stats = stats_by_bucket[name]
        drops = ", ".join(
            f"{k.split(':', 1)[1]}={v}" for k, v in sorted(stats.items()) if k.startswith("drop:")
        ) or "—"
        tokens = _token_summary([r["n_tokens"] for r in bucket_rows])
        nsfw_ratio = (
            sum(1 for r in bucket_rows if r["nsfw"]) / len(bucket_rows)
            if bucket_rows
            else 0.0
        )
        zh_ratio = (
            sum(1 for r in bucket_rows if r["lang"] == "zh") / len(bucket_rows)
            if bucket_rows
            else 0.0
        )
        lines.append(
            f"| {name} | {stats['read']} | {len(bucket_rows)} | {drops} | "
            f"{tokens['median']} | {tokens['p95']} | {nsfw_ratio:.1%} | {zh_ratio:.1%} |"
        )
    lines.append("")

    lines.append("## 划分")
    lines.append("")
    lines.append("| 划分 | 条数 | 占比 | token 中位数 | NSFW | 中文占比 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        tokens = _token_summary([r["n_tokens"] for r in part])
        nsfw_ratio = sum(1 for r in part if r["nsfw"]) / len(part) if part else 0.0
        zh_ratio = sum(1 for r in part if r["lang"] == "zh") / len(part) if part else 0.0
        lines.append(
            f"| {split} | {len(part)} | {len(part) / max(1, len(rows)):.1%} | "
            f"{tokens['median']} | {nsfw_ratio:.1%} | {zh_ratio:.1%} |"
        )
    lines.append("")

    lines.append("## 来源 × 语种 × NSFW")
    lines.append("")
    lines.append("| 来源 | zh-SFW | zh-NSFW | en-SFW | en-NSFW |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    sources = sorted({r["source"] for r in rows})
    for source in sources:
        part = [r for r in rows if r["source"] == source]
        counts = {
            ("zh", True): sum(1 for r in part if r["lang"] == "zh" and r["nsfw"]),
            ("zh", False): sum(1 for r in part if r["lang"] == "zh" and not r["nsfw"]),
            ("en", True): sum(1 for r in part if r["lang"] == "en" and r["nsfw"]),
            ("en", False): sum(1 for r in part if r["lang"] == "en" and not r["nsfw"]),
        }
        lines.append(
            f"| {source} | {counts[('zh', False)]} | {counts[('zh', True)]} | "
            f"{counts[('en', False)]} | {counts[('en', True)]} |"
        )
    lines.append("")

    official = [r for r in rows if r["bucket"] == "official"]
    if official:
        lines.append("## 官方样例（仅测试集）")
        lines.append("")
        lines.append("| id | token |")
        lines.append("| --- | ---: |")
        for row in sorted(official, key=lambda r: r["source_id"]):
            lines.append(f"| {row['source_id']} | {row['n_tokens']} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_card(path: Path, rows: list[dict], cfg: dict) -> None:
    """生成 `dataset_card.md`，包含来源表、许可与署名。

    Args:
        path: 输出路径。
        rows: 全部保留样本。
        cfg: 全局配置。

    Returns:
        无。直接写文件。
    """
    lines = ["# 数据集说明", ""]
    lines.append(
        "把 MOSS-Image 的 Qwen3.5-9B 文本编码器换成 Qwen3.5-0.8B 时，"
        "用来拟合逐 token 投影的训练语料。"
    )
    lines.append("")
    lines.append("## 构建方式")
    lines.append("")
    lines.append("```bash")
    lines.append("python src/scripts/download_models.py")
    lines.append("python -m moss_small_encoder.build --config src/configs/dataset_v1.yaml")
    lines.append("```")
    lines.append("")
    lines.append("全部来源钉死 commit sha，随机种子固定，重跑结果一致。")
    lines.append("")
    lines.append("## 来源与许可")
    lines.append("")
    lines.append("| 来源 | 条数 | 许可 | 地址 |")
    lines.append("| --- | ---: | --- | --- |")
    for source in sorted({r["source"] for r in rows}):
        count = sum(1 for r in rows if r["source"] == source)
        lines.append(
            f"| {source} | {count} | {LICENSES.get(source, '未知')} | "
            f"{SOURCE_URLS.get(source, '—')} |"
        )
    lines.append("")
    lines.append("## 必须署名")
    lines.append("")
    lines.append(
        "> 本项目使用了「水仙的AI提示词」"
        "（https://github.com/BaYue-SYJ/shuixian-prompts）的中文提示词数据，"
        "该数据按 Attribution Required License v1.0 使用。"
    )
    lines.append("")
    lines.append("## 已知限制")
    lines.append("")
    lines.append("- 超过 4096 token 的样本整条剔除（与生成侧上限一致），长 layout 覆盖偏少。")
    lines.append("- `goku` 的中文是机器翻译，存在少量混合语言句子；只用于补充语体覆盖。")
    lines.append("- layout 样本由模板确定性合成，不是真人写的提示词，坐标分布依赖切分策略。")
    lines.append("- `jtatman` / `gustavosta` 上游未标注许可，商用前需自行确认。")
    lines.append("- NSFW 标注：`jtatman` / `goku` 用源标签，其余用关键词启发式，允许误差。")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("src/configs/dataset_v1.yaml"))
    parser.add_argument("--out", type=Path, default=None, help="覆盖输出目录")
    parser.add_argument("--limit", action="append", default=[], metavar="BUCKET=N")
    parser.add_argument(
        "--max-read",
        type=int,
        default=None,
        help="每个桶最多读多少条候选，用于冒烟测试",
    )
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=None,
        help="覆盖所有 parquet 桶的 scan_rows，用于冒烟测试",
    )
    parser.add_argument(
        "--scan-bytes",
        type=int,
        default=None,
        help="覆盖所有 JSONL 桶的 scan_bytes，用于冒烟测试",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("models/tokenizer/tokenizer.json")
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["max_tokens"] = min(int(cfg.get("max_tokens", MAX_TEXT_TOKENS)), MAX_TEXT_TOKENS)
    if args.out is not None:
        cfg["out_dir"] = str(args.out)
    for override in args.limit:
        name, _, value = override.partition("=")
        for bucket in cfg["buckets"]:
            if bucket["name"] == name:
                bucket["quota"] = int(value)
                break
    if args.scan_rows is not None:
        for bucket in cfg["buckets"]:
            if "scan_rows" in bucket:
                bucket["scan_rows"] = args.scan_rows
    if args.scan_bytes is not None:
        for bucket in cfg["buckets"]:
            if "scan_bytes" in bucket:
                bucket["scan_bytes"] = args.scan_bytes

    tokenizer = MossTokenizer.from_file(args.tokenizer)
    out_dir = Path(cfg["out_dir"])
    started = time.time()

    all_rows: list[dict] = []
    stats_by_bucket: dict[str, Counter] = {}
    seen: set[str] = set()

    for bucket in tqdm(cfg["buckets"], desc="buckets", unit="bucket"):
        rows, stats = collect_bucket(
            bucket,
            cfg,
            tokenizer,
            seen,
            forced_split=bucket.get("split"),
            max_read=args.max_read,
        )
        stats_by_bucket[bucket["name"]] = stats
        all_rows.extend(rows)
        print(
            f"[{bucket['name']}] 保留 {len(rows)} / 读到 {stats['read']}"
            f" / 候选 {stats['candidates']}",
            flush=True,
        )

    all_rows.sort(key=lambda r: (r["bucket"], r["id"]))
    order = [bucket["name"] for bucket in cfg["buckets"]]

    if args.dry_run:
        print(json.dumps(
            {
                name: {
                    "kept": sum(1 for r in all_rows if r["bucket"] == name),
                    **{k: v for k, v in stats.items()},
                }
                for name, stats in stats_by_bucket.items()
            },
            ensure_ascii=False,
            indent=2,
        ))
        print(f"dry-run 完成，用时 {time.time() - started:.1f}s，未写任何文件")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "prompts.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            payload = {k: v for k, v in row.items() if k != "text"}
            payload["text"] = row["text"]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    manifest = {
        "seed": cfg["seed"],
        "max_tokens": cfg["max_tokens"],
        "splits": cfg["splits"],
        "buckets": [
            {
                "name": bucket["name"],
                "kind": bucket["kind"],
                "source": bucket.get("source"),
                "quota": bucket["quota"],
                "forced_split": bucket.get("split"),
            }
            for bucket in cfg["buckets"]
        ],
        "source_revisions": SOURCE_REVISIONS,
        "rows": len(all_rows),
        "tokens": sum(r["n_tokens"] for r in all_rows),
        "per_bucket": {
            name: {
                **dict(stats),
                "kept": sum(1 for r in all_rows if r["bucket"] == name),
            }
            for name, stats in stats_by_bucket.items()
        },
        "packages": _package_versions(),
        "output": {"prompts.jsonl": {"sha256": digest, "bytes": jsonl_path.stat().st_size}},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_stats(out_dir / "stats.md", all_rows, stats_by_bucket, order, cfg)
    write_card(out_dir / "dataset_card.md", all_rows, cfg)

    print(
        f"完成：{len(all_rows)} 条 / {manifest['tokens']:,} token，"
        f"用时 {manifest['elapsed_seconds']}s -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
