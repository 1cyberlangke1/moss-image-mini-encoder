"""下载 teacher / student 权重与分词器。

所有版本都钉死在 commit sha 上，不用 `main`：源仓库一改，别人复现出来的
数据集就不是同一份了。下完逐个校验 sha256，对不上就删掉重下。

用法：
    python scripts/download_models.py --dest models
    python scripts/download_models.py --dest models --teacher-from-local <本机文件>
    python scripts/download_models.py --dest models --check      # 只校验已有文件
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# 权重来源与校验值。改这里就等于改数据集的可复现基线，必须同步更新 models/README.md。
TEACHER = {
    "repo": "Comfy-Org/MOSS-Image",
    "revision": "530a4c94f92c35ef6818b1b035a983fe9dcf5f54",
    "path": "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors",
    "dest": "qwen3.5_9b_w4a8_gptq.safetensors",
    "bytes": 7343591088,
    "sha256": "4cc6a1543ad2061806da46f35f90cadb4801dcba246c44782cf42ec0feb6407c",
}

STUDENT = {
    "repo": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "dest_dir": "qwen3.5-0.8b",
    "files": {
        "config.json": None,
        "model.safetensors-00001-of-00001.safetensors": 1746942600,
        "model.safetensors.index.json": None,
        "tokenizer.json": None,
        "tokenizer_config.json": None,
        "vocab.json": None,
        "merges.txt": None,
        "chat_template.jinja": None,
        "preprocessor_config.json": None,
        "video_preprocessor_config.json": None,
        "LICENSE": None,
    },
}

# 分词器单独放一份，编码阶段只读这一处，避免有人换掉 student 目录时分词器跟着变。
# 注意：这里必须用真正的 sha256。tokenizer.json 不是 LFS 文件，HF 返回的 ETag
# 不是内容摘要，早先误把 ETag 当 sha256 记下来过，已被本值取代。
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """分块计算文件 sha256。

    Args:
        path: 目标文件。
        chunk: 每次读取的字节数，默认 16 MiB。

    Returns:
        小写十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    print(message, flush=True)


def place_teacher(dest_dir: Path, local: Path | None) -> Path:
    """把 teacher 放到 models/ 下，优先硬链接本机已有副本。

    Args:
        dest_dir: models/ 目录。
        local: 本机已有的同文件；None 表示从 Hugging Face 下载。

    Returns:
        落位后的路径。sha256 不匹配时抛 SystemExit。
    """
    target = dest_dir / TEACHER["dest"]

    if local is not None:
        if not local.is_file():
            raise SystemExit(f"本机 teacher 文件不存在：{local}")
        size = local.stat().st_size
        if size != TEACHER["bytes"]:
            raise SystemExit(
                f"本机 teacher 字节数不符：期望 {TEACHER['bytes']}，实际 {size}"
            )
        log(f"校验本机 teacher sha256（{size / 1024**3:.2f} GiB，约需一分钟）…")
        got = sha256_file(local)
        if got != TEACHER["sha256"]:
            raise SystemExit(f"本机 teacher sha256 不符：{got}")
        log("  sha256 一致")
        if target.exists():
            log(f"  已存在，跳过：{target}")
            return target
        try:
            os.link(local, target)
            log(f"  已硬链接（不额外占盘）：{target}")
        except OSError:
            shutil.copy2(local, target)
            log(f"  跨卷，已复制：{target}")
        return target

    if target.is_file() and target.stat().st_size == TEACHER["bytes"]:
        log(f"teacher 已存在，校验 sha256…")
        if sha256_file(target) == TEACHER["sha256"]:
            log("  sha256 一致，跳过下载")
            return target
        log("  sha256 不符，重新下载")
        target.unlink()

    log(f"下载 teacher：{TEACHER['repo']}@{TEACHER['revision'][:12]}")
    cached = hf_hub_download(
        repo_id=TEACHER["repo"],
        filename=TEACHER["path"],
        revision=TEACHER["revision"],
        local_dir=str(dest_dir),
    )
    downloaded = Path(cached)
    if downloaded != target and downloaded.is_file():
        shutil.move(str(downloaded), target)
    log("  校验 sha256…")
    if sha256_file(target) != TEACHER["sha256"]:
        target.unlink(missing_ok=True)
        raise SystemExit("teacher sha256 不符，已删除，请重试")
    log("  sha256 一致")
    return target


def fetch_student(dest_dir: Path) -> Path:
    """下载 Qwen3.5-0.8B 全部需要的文件。

    Args:
        dest_dir: models/ 目录。

    Returns:
        student 目录路径。
    """
    target_dir = dest_dir / STUDENT["dest_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_bytes in STUDENT["files"].items():
        out = target_dir / name
        if out.is_file() and (
            expected_bytes is None or out.stat().st_size == expected_bytes
        ):
            continue
        log(f"下载 student：{name}")
        # local_dir 直接落到 models/，不走「缓存再复制」：省一次 1.75 GB 的拷贝。
        hf_hub_download(
            repo_id=STUDENT["repo"],
            filename=name,
            revision=STUDENT["revision"],
            local_dir=str(target_dir),
        )
        if not out.is_file():
            raise SystemExit(f"下载后找不到 {out}")
        if expected_bytes is not None and out.stat().st_size != expected_bytes:
            raise SystemExit(f"{name} 字节数不符，期望 {expected_bytes}")
    return target_dir


def place_tokenizer(student_dir: Path, dest_dir: Path) -> Path:
    """把分词器单独复制到 models/tokenizer/。

    Args:
        student_dir: 已下载的 student 目录。
        dest_dir: models/ 目录。

    Returns:
        分词器目录路径。sha256 不符时抛 SystemExit。
    """
    tokenizer_dir = dest_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    target = tokenizer_dir / "tokenizer.json"
    source = student_dir / "tokenizer.json"
    if not source.is_file():
        raise SystemExit(f"student 目录里没有 tokenizer.json：{source}")
    shutil.copy2(source, target)
    got = sha256_file(target)
    if got != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json sha256 不符：{got}")
    log(f"分词器就位：{target}")
    return tokenizer_dir


def check(dest_dir: Path) -> int:
    """只校验已有文件，不下载。

    Args:
        dest_dir: models/ 目录。

    Returns:
        进程退出码，0 表示全部通过。
    """
    ok = True
    teacher = dest_dir / TEACHER["dest"]
    if not teacher.is_file():
        log(f"缺失 teacher：{teacher}")
        ok = False
    elif teacher.stat().st_size != TEACHER["bytes"]:
        log(f"teacher 字节数不符：{teacher.stat().st_size}")
        ok = False
    else:
        log("teacher 字节数一致，校验 sha256…")
        if sha256_file(teacher) != TEACHER["sha256"]:
            log("  sha256 不符")
            ok = False
        else:
            log("  sha256 一致")

    tokenizer = dest_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        log(f"缺失分词器：{tokenizer}")
        ok = False
    elif sha256_file(tokenizer) != TOKENIZER_SHA256:
        log("分词器 sha256 不符")
        ok = False
    else:
        log("分词器 sha256 一致")

    student_dir = dest_dir / STUDENT["dest_dir"]
    missing = [n for n in STUDENT["files"] if not (student_dir / n).is_file()]
    if missing:
        log(f"缺失 student 文件：{missing}")
        ok = False
    else:
        log("student 文件齐全")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("models"), help="权重目录")
    parser.add_argument(
        "--teacher-from-local",
        type=Path,
        default=None,
        help="本机已有的同一个 teacher 文件，校验后硬链接，省下载",
    )
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    args = parser.parse_args(argv)

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        return check(dest)

    student_dir = dest / STUDENT["dest_dir"]
    if not args.skip_teacher:
        place_teacher(dest, args.teacher_from_local)
    if not args.skip_student:
        student_dir = fetch_student(dest)
    if student_dir.is_dir() and (student_dir / "tokenizer.json").is_file():
        place_tokenizer(student_dir, dest)
    else:
        log("没有 student 目录，跳过分词器")

    return check(dest)


if __name__ == "__main__":
    sys.exit(main())
