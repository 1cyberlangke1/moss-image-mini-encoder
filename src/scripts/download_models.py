"""下载训练侧权重、生成侧底模与分词器（分组管理，逐个校验 sha256）。

三组清单（`--group` 选，`--only` 精确挑名字）喵：

- `train`（默认）：teacher 9B w4a8、Qwen3.5-0.8B/2B/4B、分词器
- `comfy`：ComfyUI 出图要的底模（DiT int8）、VAE、4 步 / 8 步 LoRA、9B 文本编码器
- `extra`：bf16 全精度 DiT 与 9B bf16 / int8 文本编码器（只在需要对比时下）

训练语料不在这里：源仓库与 revision 钉在 `src/configs/dataset_v1.yaml`，
由 `python -m moss_small_encoder.build` 按需读取，不整包下载喵。

所有版本都钉死在 commit sha 上，不用 `main`：源仓库一改，别人复现出来的
结果就不是同一份了。下完逐个校验 sha256，对不上就删掉重下。

用自己写的分段 range 下载，不用 `hf_hub_download`：实测在该网络下
`hf_hub` 的传输层会卡在 0 字节，而普通 range 请求能跑满带宽。

用法：
    python src/scripts/download_models.py --dest models                 # 训练侧
    python src/scripts/download_models.py --dest models --group comfy   # 出图侧
    python src/scripts/download_models.py --dest models --only student-4b
    python src/scripts/download_models.py --dest models --list
    python src/scripts/download_models.py --dest models --check         # 只校验
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# 让 ssl 改用系统证书库：GitHub / HF 的部分接口证书链不完整，OpenSSL 不会补链。
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from huggingface_hub import HfApi

# 生成侧与 teacher 共用同一个仓库，所以版本号只写一次喵
MOSS_REPO = "Comfy-Org/MOSS-Image"
MOSS_REVISION = "530a4c94f92c35ef6818b1b035a983fe9dcf5f54"

STUDENT_08B = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
STUDENT_2B = ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")
STUDENT_4B = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")

# 4 步 / 8 步蒸馏 LoRA。本机三份文件都用 sha256 反查确认过出处喵
LORA_4STEP = ("akhaliq/MOSS-Image-SFT-4-steps", "2ded95c37a029360245926fe72e475e30039bbfb")
LORA_SFT8 = ("lodestones/MOSS-Image-SFT-8-steps-dmd", "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f")
LORA_BASE8 = ("lodestones/MOSS-Image-Base-8-steps-dmd", "da05cb3b52899179e2bb5f7efe1058fefcbae21b")

# tokenizer.json 单独放一份，编码阶段只读这一处。
# 这个值是真 sha256；HF 的 ETag 只对 LFS 文件等于 sha256，早先误用过一次。
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TEACHER_FILE = "qwen3.5_9b_w4a8_gptq.safetensors"
STUDENT_DIR = "qwen3.5-0.8b"

WORKERS = 4
CHUNK = 1 << 20
TEACHER_9B = "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"


@dataclass(frozen=True)
class Artifact:
    """一个要落盘的对象：单文件，或整个仓库。

    Attributes:
        name: `--only` 用的注册名。
        group: `train` / `comfy` / `extra`。
        repo: Hugging Face 仓库 id。
        revision: 钉死的 commit sha。
        path: 仓库内路径；空串表示整仓库（学生的分词器、config 等一起下）。
        dest: `--dest` 下的相对路径。
        copy_to: 下游还要复制一份到的位置（分词器从学生目录复制出来）。
        note: `--list` 里显示的一句话说明。
    """

    name: str
    group: str
    repo: str
    revision: str
    path: str
    dest: str
    copy_to: str | None = None
    note: str = ""


ARTIFACTS: tuple[Artifact, ...] = (
    # ---- 训练侧 ----
    Artifact(
        "teacher-9b", "train", MOSS_REPO, MOSS_REVISION, TEACHER_9B,
        TEACHER_FILE, note="9B w4a8 文本编码器（教师，也是 ComfyUI 用的那份）",
    ),
    Artifact(
        "student-0.8b", "train", STUDENT_08B[0], STUDENT_08B[1], "",
        "qwen3.5-0.8b", copy_to="tokenizer", note="0.8B 学生全仓库 + 分词器来源",
    ),
    Artifact(
        "student-2b", "train", STUDENT_2B[0], STUDENT_2B[1], "", "qwen3.5-2b",
        note="2B 学生全仓库",
    ),
    Artifact(
        "student-4b", "train", STUDENT_4B[0], STUDENT_4B[1], "", "qwen3.5-4b",
        note="4B 学生全仓库（约 8.4 GB，两个分片）",
    ),
    # ---- 生成侧底模 ----
    Artifact(
        "dit-sft-int8", "comfy", MOSS_REPO, MOSS_REVISION,
        "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        note="SFT 底模 int8（出图默认用这份）",
    ),
    Artifact(
        "dit-base-int8", "comfy", MOSS_REPO, MOSS_REVISION,
        "diffusion_models/moss_image_base_int8_convrot.safetensors",
        "diffusion_models/moss_image_base_int8_convrot.safetensors",
        note="Base 底模 int8",
    ),
    Artifact(
        "vae-flux2-small", "comfy", MOSS_REPO, MOSS_REVISION,
        "vae/flux2-vae_small.safetensors", "vae/flux2-vae_small.safetensors",
        note="FLUX.2 VAE（小号，249.5 MB）",
    ),
    Artifact(
        "text-encoder-9b", "comfy", MOSS_REPO, MOSS_REVISION, TEACHER_9B,
        f"text_encoders/{TEACHER_FILE}", note="与 teacher 同一文件，走硬链接不占双份",
    ),
    Artifact(
        "lora-sft-4step", "comfy", LORA_4STEP[0], LORA_4STEP[1],
        "moss_image_sft_4step_v2.0.safetensors", "loras/MOSS-Image-SFT-4-steps.safetensors",
        note="SFT 4 步蒸馏 LoRA v2.0",
    ),
    Artifact(
        "lora-sft-8step", "comfy", LORA_SFT8[0], LORA_SFT8[1],
        "lora_latest.safetensors", "loras/moss_sft_8step_dmd.safetensors",
        note="SFT 8 步蒸馏 LoRA",
    ),
    Artifact(
        "lora-base-8step", "comfy", LORA_BASE8[0], LORA_BASE8[1],
        "lora_dmd2.safetensors", "loras/moss_base_8step_dmd2.safetensors",
        note="Base 8 步蒸馏 LoRA（DMD2）",
    ),
    # ---- 只在对比时下 ----
    Artifact(
        "dit-sft-bf16", "extra", MOSS_REPO, MOSS_REVISION,
        "diffusion_models/moss_image_sft_bf16.safetensors",
        "diffusion_models/moss_image_sft_bf16.safetensors",
        note="SFT 底模 bf16（10.2 GB，8 GB 卡跑不动）",
    ),
    Artifact(
        "text-encoder-9b-bf16", "extra", MOSS_REPO, MOSS_REVISION,
        "text_encoders/qwen3.5_9b_bf16.safetensors",
        "text_encoders/qwen3.5_9b_bf16.safetensors",
        note="9B 文本编码器 bf16（19.3 GB）",
    ),
    Artifact(
        "text-encoder-9b-int8", "extra", MOSS_REPO, MOSS_REVISION,
        "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        note="9B 文本编码器 int8（9.7 GB）",
    ),
)

_META_CACHE: dict[tuple[str, str], dict[str, tuple[int, str | None]]] = {}


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
    return f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


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
                raise OSError(f"区间长度不符：期望 {end - start + 1}，收到 {len(data)}")
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
        每段直接写进预分配好的不同偏移，不产生额外临时文件。
        断点续传没做：单文件十几分钟，重试整段比处理半截文件简单喵。
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
    """取仓库在指定 revision 下每个文件的字节数与 LFS sha256（按仓库缓存）。

    Args:
        repo: 仓库 id。
        revision: commit sha。

    Returns:
        路径 -> (字节数, sha256 或 None)。非 LFS 文件没有 sha256。

    Notes:
        走 API 只读元信息，不下载内容；同一仓库在一次运行里只查一次喵。
    """
    key = (repo, revision)
    if key not in _META_CACHE:
        info = HfApi().model_info(repo_id=repo, revision=revision, files_metadata=True)
        _META_CACHE[key] = {
            sibling.rfilename: (sibling.size or 0, (sibling.lfs or {}).get("sha256"))
            for sibling in (info.siblings or [])
        }
    return _META_CACHE[key]


def ensure_file(artifact: Artifact, dest_root: Path, link_from: Path | None) -> None:
    """确保一个单文件就位：已有就校验，`link_from` 给定时硬链接而不是下载。

    Args:
        artifact: 注册表条目（`path` 必须是具体文件）。
        dest_root: `--dest` 目录。
        link_from: 本机已有的同一份文件；None 表示从 Hugging Face 下。

    Returns:
        无，失败抛 SystemExit。

    Raises:
        SystemExit: 仓库里没有该路径、字节数或 sha256 不符。
    """
    meta = file_meta(artifact.repo, artifact.revision).get(artifact.path)
    if meta is None:
        raise SystemExit(f"{artifact.repo} 里找不到 {artifact.path}")
    size, sha = meta
    target = dest_root / artifact.dest

    if link_from is not None and link_from.is_file():
        if link_from.stat().st_size != size:
            raise SystemExit(
                f"本机文件字节数不符：期望 {size}，实际 {link_from.stat().st_size}"
            )
        log(f"[{artifact.name}] 校验本机文件 sha256（{size / 1e9:.2f} GB）…")
        if sha and sha256_file(link_from) != sha:
            raise SystemExit(f"本机文件 sha256 不符：{link_from}")
        if target.exists():
            log(f"  已存在，跳过：{target}")
            return
        try:
            os.link(link_from, target)
            log(f"  已硬链接（不额外占盘）：{target}")
        except OSError:
            shutil.copy2(link_from, target)
            log(f"  跨卷，已复制：{target}")
        return

    if target.is_file() and target.stat().st_size == size:
        if sha is None or sha256_file(target) == sha:
            log(f"[{artifact.name}] 已就位：{target}")
            return
        log(f"[{artifact.name}] sha256 不符，重新下载")
        target.unlink()

    log(f"[{artifact.name}] 下载 {artifact.repo}@{artifact.revision[:12]}/{artifact.path}")
    download(resolve_url(artifact.repo, artifact.path, artifact.revision), target, size)
    if sha is not None and sha256_file(target) != sha:
        target.unlink(missing_ok=True)
        raise SystemExit(f"[{artifact.name}] sha256 不符，已删除，请重试")
    log(f"[{artifact.name}] sha256 一致")


def ensure_repo(artifact: Artifact, dest_root: Path) -> None:
    """下载整个仓库（学生的 config / 权重分片 / 分词器一起取）。

    Args:
        artifact: 注册表条目（`path` 为空串）。
        dest_root: `--dest` 目录。

    Returns:
        无，失败抛 SystemExit。

    Notes:
        逐文件比对大小与 sha256，已存在的跳过，所以中断后重跑只会补缺的喵。
    """
    target_dir = dest_root / artifact.dest
    meta = file_meta(artifact.repo, artifact.revision)
    if not meta:
        raise SystemExit(f"{artifact.repo} 元信息为空")

    for name in sorted(meta):
        size, sha = meta[name]
        if name.startswith(".") or size == 0:
            continue
        out = target_dir / name
        if out.is_file() and out.stat().st_size == size:
            if sha is None or sha256_file(out) == sha:
                continue
        log(f"[{artifact.name}] 下载 {name}  ({size / 1e6:.1f} MB)")
        download(resolve_url(artifact.repo, name, artifact.revision), out, size)
        if sha is not None and sha256_file(out) != sha:
            out.unlink(missing_ok=True)
            raise SystemExit(f"[{artifact.name}] {name} sha256 不符，已删除")

    if artifact.copy_to:
        place_tokenizer(target_dir, dest_root / artifact.copy_to)


def place_tokenizer(student_dir: Path, tokenizer_dir: Path) -> Path:
    """把分词器单独复制一份到 `models/tokenizer/`。

    Args:
        student_dir: 已下载的学生目录。
        tokenizer_dir: 目标目录。

    Returns:
        目标文件路径。sha256 不符时抛 SystemExit。

    Notes:
        编码阶段只读这一处，避免依赖某个学生目录还在不在喵。
    """
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    target = tokenizer_dir / "tokenizer.json"
    source = student_dir / "tokenizer.json"
    if not source.is_file():
        raise SystemExit(f"学生目录里没有 tokenizer.json：{source}")
    if not target.is_file() or sha256_file(target) != TOKENIZER_SHA256:
        shutil.copy2(source, target)
    got = sha256_file(target)
    if got != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json sha256 不符：{got}")
    log(f"分词器就位：{target}")
    return target


def check_artifact(artifact: Artifact, dest_root: Path) -> bool:
    """校验一个条目是否齐全（不下载）。

    Args:
        artifact: 注册表条目。
        dest_root: `--dest` 目录。

    Returns:
        True 表示文件齐全且 sha256 一致。
    """
    meta = file_meta(artifact.repo, artifact.revision)
    if artifact.path:
        names = {artifact.path: artifact.dest}
    else:
        names = {
            name: f"{artifact.dest}/{name}"
            for name in meta
            if not name.startswith(".") and meta[name][0] > 0
        }

    ok = True
    for repo_path, relative in names.items():
        size, sha = meta[repo_path]
        target = dest_root / relative
        if not target.is_file():
            log(f"[{artifact.name}] 缺失 {relative}")
            ok = False
        elif target.stat().st_size != size:
            log(f"[{artifact.name}] 字节数不符 {relative}")
            ok = False
        elif sha is not None and sha256_file(target) != sha:
            log(f"[{artifact.name}] sha256 不符 {relative}")
            ok = False
    if artifact.copy_to:
        tokenizer = dest_root / artifact.copy_to / "tokenizer.json"
        if not tokenizer.is_file() or sha256_file(tokenizer) != TOKENIZER_SHA256:
            log(f"[{artifact.name}] 分词器缺失或不符：{tokenizer}")
            ok = False
    if ok:
        log(f"[{artifact.name}] 通过")
    return ok


def select(group: str, only: list[str] | None) -> list[Artifact]:
    """按组或名字挑出要处理的条目。

    Args:
        group: `train` / `comfy` / `extra` / `all`。
        only: 逗号分隔的注册名；给定时忽略 `group`。

    Returns:
        选中的条目列表。

    Raises:
        SystemExit: 名字不存在。
    """
    if only:
        known = {artifact.name: artifact for artifact in ARTIFACTS}
        missing = [name for name in only if name not in known]
        if missing:
            raise SystemExit(f"未注册的名字 {missing}；可用：{sorted(known)}")
        return [known[name] for name in only]
    if group == "all":
        return list(ARTIFACTS)
    picked = [artifact for artifact in ARTIFACTS if artifact.group == group]
    if not picked:
        raise SystemExit(f"组 {group!r} 里没有条目")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("models"), help="权重根目录")
    parser.add_argument(
        "--group",
        default="train",
        choices=["train", "comfy", "extra", "all"],
        help="train=训练侧（默认），comfy=出图底模，extra=对比用大文件喵",
    )
    parser.add_argument("--only", default="", help="逗号分隔的注册名，优先于 --group")
    parser.add_argument("--list", action="store_true", help="只打印清单")
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    parser.add_argument(
        "--teacher-from-local",
        type=Path,
        default=None,
        help="本机已有的 9B w4a8 文件，校验后硬链接，省下载",
    )
    args = parser.parse_args(argv)

    if args.list:
        for artifact in ARTIFACTS:
            log(
                f"  {artifact.group:<6} {artifact.name:<22} "
                f"{artifact.dest:<52} {artifact.note}"
            )
        return 0

    only = [name.strip() for name in args.only.split(",") if name.strip()]
    picked = select(args.group, only or None)
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        return 0 if all(check_artifact(a, dest) for a in picked) else 1

    for artifact in picked:
        # teacher 与 ComfyUI 的 9B 文本编码器是同一个文件，两处都允许走本机硬链接喵
        link_from = args.teacher_from_local if artifact.path == TEACHER_9B else None
        if artifact.path:
            ensure_file(artifact, dest, link_from)
        else:
            ensure_repo(artifact, dest)

    return 0 if all(check_artifact(a, dest) for a in picked) else 1


if __name__ == "__main__":
    sys.exit(main())
