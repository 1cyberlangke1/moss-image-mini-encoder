"""检查 ComfyUI 用的分词器与 Hugging Face 官方 tokenizer.json 是否等价。

为什么必须查：teacher/student 的特征是按 token 位置对齐的，而生成时实际用的是
ComfyUI 自带的那份词表（`qwen35_tokenizer/vocab.json` + `merges.txt`）。
两份文件的字节数不同（HF 6722759 / 3353259，ComfyUI 6970804 / 3600846），
如果它们的 token id 不一致，我们算出来的特征就和生成时编码的根本不是同一串 token。

本工具做两层检查：

1. **结构比对**：词表键集合、id 映射、merges 顺序是否完全一致。
   BPE 在给定 vocab + merges + 同一套 pipeline 时是确定性的，
   所以这一层一致就足以说明 id 一致。
2. **端到端比对**（`--end-to-end`）：用 ComfyUI 的 vocab/merges 重建一个
   tokenizers 对象，复制 HF 的 pipeline，在故意刁钻的语料上逐条比对 id。

用法：
    python src/scripts/parity_tokenizer.py \
        --hf models/tokenizer/tokenizer.json \
        --comfy <ComfyUI>/comfy/text_encoders/qwen35_tokenizer \
        --end-to-end

本脚本只读外部文件，不 import ComfyUI 的任何模块。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 故意挑容易暴露差异的样本：重音、中日韩、emoji 组合、特殊 token、
# 数字与科学计数、多空格与制表、换行，以及 MOSS 自己的 layout JSON。
HOSTILE = [
    "a red ball falls onto the ground",
    "Nicolas, comment vas-tu ? Ça va très bien, merci beaucoup.",
    "Où est passé l'œuf de Noël ? Straße, mañana, ångström.",
    "日本語のテキスト、中文文本，한국어 텍스트",
    "Здравствуйте, как дела? Γειά σου κόσμε. مرحبا بالعالم",
    "emoji: 🎬 🎥 📽️ 🍿 👩‍🚀 👨‍👩‍👧‍👦 🏳️‍🌈",
    "<|im_start|> <|im_end|> <|endoftext|> <think>",
    "1234567890 3.14159 1e-9 0x1F 100% $50 €30 £20",
    "   multiple    spaces\tand\ttabs   ",
    "line one\nline two\n\nline four",
    "MiXeD CaSe AnD Punctuation!!! ??? ... --- ___ ***",
    "a" * 300,
    "",
    '{"summary":{"caption":"极简海报"},"items":[{"label":"text","caption":"标题",'
    '"text":"把日常，过成诗。","bbox":[-288,-592,352,32]}],'
    '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}',
]


def load_hf_spec(path: Path) -> dict:
    """读取 tokenizer.json 的原始 JSON。

    Args:
        path: tokenizer.json 路径。

    Returns:
        解析后的字典。
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_comfy_vocab(path: Path) -> dict[str, int]:
    """读取 ComfyUI 的 vocab.json。

    Args:
        path: vocab.json 路径。

    Returns:
        token 字符串到 id 的映射。
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_comfy_merges(path: Path) -> list[str]:
    """读取 merges.txt，返回「左 右」形式的合并规则列表。

    Args:
        path: merges.txt 路径。

    Returns:
        合并规则列表，已去掉空行。首行若是 `#version:` 注释会被跳过。
    """
    rules = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            rules.append(line)
    return rules


def compare_structure(hf: dict, vocab: dict[str, int], merges: list[str]) -> list[str]:
    """比对 HF tokenizer.json 与 ComfyUI 词表结构。

    Args:
        hf: HF tokenizer.json 的解析结果。
        vocab: ComfyUI 的 vocab.json。
        merges: ComfyUI 的 merges.txt 规则。

    Returns:
        差异描述列表；空列表表示结构完全一致。
    """
    problems: list[str] = []
    model = hf.get("model") or {}
    hf_vocab: dict[str, int] = model.get("vocab") or {}
    hf_merges = model.get("merges") or []

    # merges 在 tokenizer.json 里可能是 "a b" 字符串，也可能是 ["a","b"] 数组。
    normalized = [
        m if isinstance(m, str) else " ".join(m) for m in hf_merges
    ]

    if len(hf_vocab) != len(vocab):
        problems.append(f"词表大小不同：HF {len(hf_vocab)} vs ComfyUI {len(vocab)}")

    only_hf = set(hf_vocab) - set(vocab)
    only_comfy = set(vocab) - set(hf_vocab)
    if only_hf:
        problems.append(f"仅 HF 有的 token {len(only_hf)} 个，例如 {sorted(only_hf)[:5]}")
    if only_comfy:
        problems.append(
            f"仅 ComfyUI 有的 token {len(only_comfy)} 个，例如 {sorted(only_comfy)[:5]}"
        )

    id_mismatch = [
        token
        for token in set(hf_vocab) & set(vocab)
        if hf_vocab[token] != vocab[token]
    ]
    if id_mismatch:
        problems.append(
            f"{len(id_mismatch)} 个 token 的 id 不同，例如 "
            f"{[(t, hf_vocab[t], vocab[t]) for t in sorted(id_mismatch)[:5]]}"
        )

    if normalized != merges:
        problems.append(f"merges 不同：HF {len(normalized)} 条 vs ComfyUI {len(merges)} 条")
        for index, (a, b) in enumerate(zip(normalized, merges)):
            if a != b:
                problems.append(f"  首个差异在第 {index} 条：HF {a!r} vs ComfyUI {b!r}")
                break

    return problems


def build_from_comfy(hf_spec: dict, vocab: dict[str, int], merges: list[str]):
    """用 ComfyUI 的词表重建一个 tokenizer，管线配置沿用 HF 的。

    Args:
        hf_spec: HF tokenizer.json 解析结果。
        vocab: ComfyUI vocab。
        merges: ComfyUI merges。
        hf_path 不需要，管线对象直接从序列化后的 JSON 反建。

    Returns:
        `tokenizers.Tokenizer` 实例。

    Notes:
        直接把 HF 的 normalizer / pre_tokenizer / decoder / post_processor
        反序列化后挂到新对象上，保证两边只差「词表」这一个变量。
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors

    model_spec = dict(hf_spec.get("model") or {})
    bpe = models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token=model_spec.get("unk_token"),
        fuse_unk=bool(model_spec.get("fuse_unk", False)),
        byte_fallback=bool(model_spec.get("byte_fallback", False)),
        ignore_merges=bool(model_spec.get("ignore_merges", False)),
    )
    tokenizer = Tokenizer(bpe)

    if hf_spec.get("normalizer"):
        tokenizer.normalizer = normalizers.Normalizer.from_str(
            json.dumps(hf_spec["normalizer"])
        )
    if hf_spec.get("pre_tokenizer"):
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.from_str(
            json.dumps(hf_spec["pre_tokenizer"])
        )
    if hf_spec.get("decoder"):
        tokenizer.decoder = decoders.Decoder.from_str(json.dumps(hf_spec["decoder"]))
    if hf_spec.get("post_processor"):
        tokenizer.post_processor = processors.PostProcessor.from_str(
            json.dumps(hf_spec["post_processor"])
        )
    for added in hf_spec.get("added_tokens") or []:
        tokenizer.add_special_tokens([added["content"]])
    return tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True, help="HF tokenizer.json")
    parser.add_argument(
        "--comfy",
        type=Path,
        required=True,
        help="ComfyUI 的 qwen35_tokenizer 目录（含 vocab.json 与 merges.txt）",
    )
    parser.add_argument("--end-to-end", action="store_true", help="再做逐条 id 比对")
    args = parser.parse_args(argv)

    vocab_path = args.comfy / "vocab.json"
    merges_path = args.comfy / "merges.txt"
    for path in (args.hf, vocab_path, merges_path):
        if not path.is_file():
            print(f"找不到文件：{path}", file=sys.stderr)
            return 2

    hf_spec = load_hf_spec(args.hf)
    vocab = load_comfy_vocab(vocab_path)
    merges = load_comfy_merges(merges_path)

    print(f"HF tokenizer.json : {args.hf}  ({args.hf.stat().st_size} 字节)")
    print(f"ComfyUI vocab.json: {vocab_path}  ({vocab_path.stat().st_size} 字节)")
    print(f"ComfyUI merges.txt: {merges_path}  ({merges_path.stat().st_size} 字节)")
    print()

    problems = compare_structure(hf_spec, vocab, merges)
    if problems:
        print("结构比对：不一致")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("结构比对：词表键、id 映射、merges 顺序完全一致")

    if not args.end_to_end:
        return 0 if not problems else 1

    print()
    from tokenizers import Tokenizer

    hf_tokenizer = Tokenizer.from_file(str(args.hf))
    comfy_tokenizer = build_from_comfy(hf_spec, vocab, merges)

    mismatches = 0
    total_tokens = 0
    for text in HOSTILE:
        a = hf_tokenizer.encode(text, add_special_tokens=False).ids
        b = comfy_tokenizer.encode(text, add_special_tokens=False).ids
        total_tokens += len(a)
        if a != b:
            mismatches += 1
            print(f"不一致：{text[:60]!r}")
            print(f"  HF     : {a[:20]}")
            print(f"  ComfyUI: {b[:20]}")

    print(f"端到端：{len(HOSTILE)} 条文本 / {total_tokens} token，{mismatches} 条不一致")
    return 0 if mismatches == 0 and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
