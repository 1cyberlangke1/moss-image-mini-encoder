"""把知识桶、硬 case、老桶合并成一份训练语料喵。

这份语料的目的是给小编码器注入「名字→作品→外观」的绑定，
不是普通监督学习，所以默认**不做 train/val/test 划分**：
按记录哈希切分会把同一个角色的部分变体抽到 val/test，
等于一部分知识样本没进训练喵。

需要留出集时用 `--split` 显式打开，比例形如 `0.7,0.15,0.15`，
按记录 id 的 sha1 确定性分桶，同一 id 永远落在同一侧喵。

用法:
    python -m moss_small_encoder.merge --out dataset/corpus
    python -m moss_small_encoder.merge --out dataset/corpus --split 0.7,0.15,0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCES = (
    ("kb", Path("dataset/kb/prompts.jsonl")),
    ("hard_cases", Path("dataset/hard_cases/v1/prompts.jsonl")),
    ("replay", Path("dataset/prompts.jsonl")),
)


def rate_of(record_id: str) -> float:
    """把 id 映射到 `[0,1)` 的确定性小数喵。

    输入: 记录 id
    输出: 0~1 的浮点数
    预期: 同一 id 永远同一个值，用来做确定性划分喵

    Notes:
        取 sha1 的**后** 8 位。取前 8 位会和 `build.split_of` 的
        前 8 位撞在一起，抽出来的子集全部落在同一侧喵。
    """
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
    return int(digest[8:16], 16) / 0xFFFFFFFF


def split_of(record_id: str, ratios: tuple[float, float, float]) -> str:
    """按 id 的确定性小数把记录分到 train/val/test喵。

    输入: 记录 id、三段比例
    输出: `"train"` / `"val"` / `"test"`
    预期: 三方比例之和为 1，越界时归到最后一个桶喵
    """
    share = rate_of(record_id)
    train, val, _ = ratios
    if share < train:
        return "train"
    if share < train + val:
        return "val"
    return "test"


def load(path: Path, tag: str) -> list[dict]:
    """读一份 prompts.jsonl 并打上来源标签喵。

    输入: 路径、来源标签
    输出: 记录列表（原地加 `origin` 字段）
    预期: 文件不存在时返回空列表并提示喵
    """
    if not path.exists():
        print(f"  跳过 {tag}: {path} 不存在", flush=True)
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["origin"] = tag
            rows.append(row)
    tokens = sum(int(r["n_tokens"]) for r in rows)
    print(f"  {tag:11s} {len(rows):>7} 条  {tokens:>10} token  <- {path}", flush=True)
    return rows


def write(rows: list[dict], out: Path) -> None:
    """写 prompts.jsonl 与 manifest.json喵。

    输入: 记录列表、输出目录
    输出: 无（落盘两个文件）
    预期: 条数与 token 数与记录一致喵
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompts.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokens = sum(int(r["n_tokens"]) for r in rows)
    manifest = {
        "rows": len(rows),
        "tokens": tokens,
        "by_origin": dict(Counter(r["origin"] for r in rows)),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_bucket": dict(Counter(r["bucket"] for r in rows)),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"  -> {path}  {len(rows):,} 条 / {tokens:,} token", flush=True)
    print(f"     origin {manifest['by_origin']}", flush=True)
    print(f"     split  {manifest['by_split']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """命令行入口喵。"""
    parser = argparse.ArgumentParser(description="合并训练语料喵")
    parser.add_argument("--out", type=Path, default=Path("dataset/corpus"))
    parser.add_argument(
        "--split",
        default="",
        help="可选：train,val,test 三段比例，如 0.7,0.15,0.15；留空表示全部进 train",
    )
    args = parser.parse_args(argv)

    ratios: tuple[float, float, float] | None = None
    if args.split:
        parts = [float(v) for v in args.split.split(",")]
        if len(parts) != 3:
            raise SystemExit("--split 需要三个比例，例如 0.7,0.15,0.15")
        ratios = (parts[0], parts[1], parts[2])

    print("读取来源:", flush=True)
    rows: list[dict] = []
    for tag, path in SOURCES:
        rows.extend(load(path, tag))

    seen: set[str] = set()
    unique: list[dict] = []
    dup = 0
    for row in rows:
        if row["id"] in seen:
            dup += 1
            continue
        seen.add(row["id"])
        unique.append(row)

    if ratios is None:
        for row in unique:
            row["split"] = "train"
        print("  未给 --split，全部进 train", flush=True)
    else:
        for row in unique:
            row["split"] = split_of(row["id"], ratios)

    if dup:
        print(f"  去重丢掉 {dup} 条重复 id", flush=True)

    write(unique, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
