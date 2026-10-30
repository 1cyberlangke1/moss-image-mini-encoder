"""下载 teacher / student 权重与分词器。

所有版本都钉死在 commit sha 上，不用 `main`：源仓库一改，别人复现出来的
数据集就不是同一份了。下完逐个校验 sha256，对不上就删掉重下。

用自己写的分段 range 下载，不用 `hf_hub_download`：实测在该网络下
`hf_hub` 的传输层会卡在 0 字节，而普通 range 请求能跑满带宽。

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
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 让 ssl 改用系统证书库：GitHub / HF 的部分接口证书链不完整，OpenSSL 不会补链。
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from huggingface_hub import HfApi

TEACHER_REPO = "Comfy-Org/MOSS-Image"
TEACHER_REVISION = "530a4c94f92c35ef6818b1b035a983fe9dcf5f54"
TEACHER_PATH = "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"
TEACHER_FILE = "qwen3.5_9b_w4a8_gptq.safetensors"

STUDENT_REPO = "Qwen/Qwen3.5-0.8B"
STUDENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
STUDENT_DIR = "qwen3.5-0.8b"

# tokenizer.json 单独放一份，编码阶段只读这一处。
# 这个值是真 sha256；HF 的 ETag 只对 LFS 文件等于 sha256，早先误用过一次。
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"

WORKERS = 4
CHUNK = 1 << 20


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """分块计算文件 sha256。

    Args:
        path: 目标文件。
        chunk: 每次读取的字节数。

    Returns:
        小写十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_url(repo: str, path: str, revision: str) -> str:
    """拼出 Hugging Face 的直链。

    Args:
        repo: 仓库 id。
        path: 仓库内路径。
        revision: commit sha。

    Returns:
        `https://huggingface.co/<repo>/resolve/<revision>/<path>`。
    """
    return f"https://huggingface.co/{repo}/resolve/{revision}/{quote(path)}"


def quote(path: str) -> str:
    """对路径做 URL 编码，保留斜杠。

    Args:
        path: 仓库内相对路径。

    Returns:
        编码后的路径。
    """
    import urllib.parse

    return urllib.parse.quote(path)


def _fetch(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """下载一个字节区间，失败重试。

    Args:
        url: 直链。
        start: 起始字节（含）。
        end: 结束字节（含）。
        retries: 重试次数。

    Returns:
        该区间的字节内容。
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url)
            request.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != end - start + 1:
                raise IOError(f"区间长度不符：期望 {end - start + 1}，收到 {len(data)}")
            return data
        except Exception as error:  # noqa: BLE001 - 网络异常的形态太多，统一重试
            last = error
            time.sleep(min(2**attempt, 20))
    raise SystemExit(f"下载失败 {url} [{start}-{end}]：{last}")


def download(url: str, dest: Path, size: int, workers: int = WORKERS) -> None:
    """分段并行下载一个文件。

    Args:
        url: 直链。
        dest: 落地路径。
        size: 期望字节数，用于切分区间与校验。
        workers: 并行段数。

    Returns:
        无。失败时抛 SystemExit。

    Notes:
        每段直接写进预分配好的文件的不同偏移，不产生额外临时文件。
        断点续传没做：当前网络下单文件十几分钟，重试整段比处理半截文件简单。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with part.open("wb") as handle:
        handle.truncate(size)

    step = (size + workers - 1) // workers
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(workers)]
    done = [0] * workers
    lock = threading.Lock()
    started = time.time()
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(2.0):
            total = sum(done)
            elapsed = max(1e-6, time.time() - started)
            percent = 100.0 * total / size
            rate = total / 1e6 / elapsed
            print(
                f"\r    {percent:5.1f}%  {total / 1e6:7.1f}/{size / 1e6:.1f} MB  "
                f"{rate:5.2f} MB/s",
                end="",
                flush=True,
            )

    def worker(index: int) -> None:
        start, end = ranges[index]
        with part.open("r+b") as handle:
            cursor = start
            while cursor <= end:
                block_end = min(end, cursor + CHUNK - 1)
                data = _fetch(url, cursor, block_end)
                handle.seek(cursor)
                handle.write(data)
                cursor = block_end + 1
                with lock:
                    done[index] += len(data)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(worker, i) for i in range(workers)]:
                future.result()
    finally:
        stop.set()
        reporter.join(timeout=1)
    print()

    if part.stat().st_size != size:
        part.unlink(missing_ok=True)
        raise SystemExit(f"下载后大小不符：{part.stat().st_size} != {size}")
    part.replace(dest)
    elapsed = time.time() - started
    log(f"    完成 {size / 1e6:.1f} MB，{elapsed:.0f}s，{size / 1e6 / elapsed:.2f} MB/s")


def file_meta(repo: str, revision: str) -> dict[str, tuple[int, str | None]]:
    """取仓库在指定 revision 下每个文件的字节数与 LFS sha256。

    Args:
        repo: 仓库 id。
        revision: commit sha。

    Returns:
        路径 -> (字节数, sha256 或 None)。非 LFS 文件没有 sha256。

    Notes:
        走 API 只读元信息，不下载内容。
    """
    api = HfApi()
    info = api.model_info(repo_id=repo, revision=revision, files_metadata=True)
    return {
        sibling.rfilename: (sibling.size or 0, (sibling.lfs or {}).get("sha256"))
        for sibling in (info.siblings or [])
    }


def place_teacher(dest_dir: Path, local: Path | None) -> Path:
    """把 teacher 放到 models/ 下，优先硬链接本机已有副本。

    Args:
        dest_dir: models/ 目录。
        local: 本机已有的同文件；None 表示从 Hugging Face 下载。

    Returns:
        落位后的路径。sha256 不匹配时抛 SystemExit。
    """
    target = dest_dir / TEACHER_FILE
    meta = file_meta(TEACHER_REPO, TEACHER_REVISION).get(TEACHER_PATH)
    if meta is None:
        raise SystemExit(f"仓库里找不到 {TEACHER_PATH}")
    size, sha = meta
    if sha is None:
        raise SystemExit("teacher 应该是 LFS 文件，但没取到 sha256")

    if local is not None:
        if not local.is_file():
            raise SystemExit(f"本机 teacher 文件不存在：{local}")
        if local.stat().st_size != size:
            raise SystemExit(f"本机 teacher 字节数不符：期望 {size}，实际 {local.stat().st_size}")
        log(f"校验本机 teacher sha256（{size / 1024**3:.2f} GiB，约需一分钟）…")
        got = sha256_file(local)
        if got != sha:
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

    if target.is_file() and target.stat().st_size == size:
        log("teacher 已存在，校验 sha256…")
        if sha256_file(target) == sha:
            log("  sha256 一致，跳过下载")
            return target
        log("  sha256 不符，重新下载")
        target.unlink()

    log(f"下载 teacher：{TEACHER_REPO}@{TEACHER_REVISION[:12]}")
    download(resolve_url(TEACHER_REPO, TEACHER_PATH, TEACHER_REVISION), target, size)
    log("  校验 sha256…")
    if sha256_file(target) != sha:
        target.unlink(missing_ok=True)
        raise SystemExit("teacher sha256 不符，已删除，请重试")
    log("  sha256 一致")
    return target


def fetch_student(dest_dir: Path) -> Path:
    """下载 Qwen3.5-0.8B 的全部文件。

    Args:
        dest_dir: models/ 目录。

    Returns:
        student 目录路径。
    """
    target_dir = dest_dir / STUDENT_DIR
    meta = file_meta(STUDENT_REPO, STUDENT_REVISION)
    if not meta:
        raise SystemExit("student 仓库元信息为空")
    for name in sorted(meta):
        size, sha = meta[name]
        if name.startswith(".") or size == 0:
            continue
        out = target_dir / name
        if out.is_file() and out.stat().st_size == size:
            if sha is None or sha256_file(out) == sha:
                continue
        log(f"下载 student：{name}  ({size / 1e6:.1f} MB)")
        download(resolve_url(STUDENT_REPO, name, STUDENT_REVISION), out, size)
        if sha is not None and sha256_file(out) != sha:
            out.unlink(missing_ok=True)
            raise SystemExit(f"{name} sha256 不符，已删除")
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
    if not target.is_file() or sha256_file(target) != TOKENIZER_SHA256:
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
    teacher = dest_dir / TEACHER_FILE
    expectation = file_meta(TEACHER_REPO, TEACHER_REVISION).get(TEACHER_PATH)
    if not teacher.is_file():
        log(f"缺失 teacher：{teacher}")
        ok = False
    elif expectation and teacher.stat().st_size != expectation[0]:
        log(f"teacher 字节数不符：{teacher.stat().st_size}")
        ok = False
    elif expectation and expectation[1] and sha256_file(teacher) != expectation[1]:
        log("teacher sha256 不符")
        ok = False
    else:
        log("teacher 校验通过")

    tokenizer = dest_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        log(f"缺失分词器：{tokenizer}")
        ok = False
    elif sha256_file(tokenizer) != TOKENIZER_SHA256:
        log("分词器 sha256 不符")
        ok = False
    else:
        log("分词器校验通过")

    student_meta = file_meta(STUDENT_REPO, STUDENT_REVISION)
    missing = [
        name
        for name, (size, _) in student_meta.items()
        if not name.startswith(".") and size > 0
        and not (dest_dir / STUDENT_DIR / name).is_file()
    ]
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

    student_dir = dest / STUDENT_DIR
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
