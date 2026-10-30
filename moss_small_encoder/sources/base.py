"""数据源共用的记录结构、远程读取与 HTTP 工具。

远程读取只取需要的那一小部分，不整包下载：
- parquet 用 HTTP range 打开，只读指定列 + 均匀散布的行组；
- jsonl 用字节区间抽样，读若干等距片段而不是整个文件。

这样「从 89 万行里取 1.2 万行」只用传输几 MB，而不是 200 MB。
"""

from __future__ import annotations

import fnmatch
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

# 让标准库 ssl 改用操作系统证书库（Windows 上是 Schannel 那套）。
# 原因：GitHub 某些接口返回的证书链不含中间证书，OpenSSL 不会自动补链，
# 而系统证书库会，所以不注入时 urllib 对 api.github.com 会报
# `CERTIFICATE_VERIFY_FAILED: unable to get issuer certificate`，
# 但 curl 和 huggingface_hub 却正常。truststore 不用改任何调用点。
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # 没装也能跑，只是 GitHub 那两个源可能连不上
    pass

_USER_AGENT = "mini-moss-image-smaller-encoder/0.1"
_FS: HfFileSystem | None = None


@dataclass
class Record:
    """一条待入库的提示词。

    Attributes:
        text: 源里的原始提示词正文（尚未按桶清洗）。
        source: 数据源短名，写进 `prompts.jsonl` 与 `manifest.json`。
        source_id: 源内稳定标识（行号、url 或原始 id），用于回溯。
        lang: `"zh"` / `"en"`；由构建流程用 `clean.detect_lang` 填。
        nsfw: True/False/None。None 表示源没有标注，由启发式判定。
        kind: `"positive"`（文生图提示词）或 `"negative"`（负面提示词）。
        meta: 源内附加信息，例如 goku 的 safety_rating。
    """

    text: str
    source: str
    source_id: str
    lang: str = "en"
    nsfw: bool | None = None
    kind: str = "positive"
    meta: dict = field(default_factory=dict)


def fs() -> HfFileSystem:
    """返回共享的 HfFileSystem 实例。

    Args:
        无。

    Returns:
        进程内单例，避免每次读文件都重新建连接。
    """
    global _FS
    if _FS is None:
        _FS = HfFileSystem()
    return _FS


def list_files(repo: str, revision: str, pattern: str, kind: str = "dataset") -> list[str]:
    """列出仓库内在某 revision 下匹配通配符的文件。

    Args:
        repo: 仓库 id。
        revision: 钉死的 commit sha。
        pattern: fnmatch 模式，例如 `data/*.parquet`。
        kind: `"dataset"` 或 `"model"`。

    Returns:
        排序后的文件相对路径列表。

    Notes:
        只是一次 API 调用，不下载内容。
    """
    api = HfApi()
    names = [
        name
        for name in api.list_repo_files(repo_id=repo, revision=revision, repo_type=kind)
        if fnmatch.fnmatch(name, pattern)
    ]
    if not names:
        raise FileNotFoundError(f"{repo}@{revision[:8]} 里没有匹配 {pattern} 的文件")
    return sorted(names)


def _remote_path(repo: str, name: str, kind: str) -> str:
    prefix = "datasets/" if kind == "dataset" else "models/"
    return f"{prefix}{repo}/{name}"


def parquet_groups(
    repo: str,
    revision: str,
    pattern: str,
    columns: list[str],
    scan_rows: int | None = None,
    kind: str = "dataset",
) -> Iterator[tuple[str, int, pa.Table]]:
    """流式读取 parquet 的选定行组，只传需要的列。

    Args:
        repo: 仓库 id。
        revision: 钉死的 commit sha。
        pattern: 文件通配符。
        columns: 需要的列名。
        scan_rows: 期望读取的总行数上限；None 表示读全部行组。
        kind: `"dataset"` 或 `"model"`。

    Yields:
        `(文件名, 该行组起始行号, Table)`。

    Notes:
        行组在文件内**等距抽取**而不是取开头：数据集常按时间或 id 排序，
        只取前 N 行会让样本偏离整体分布。等距抽取让覆盖更均匀，且完全确定。
    """
    names = list_files(repo, revision, pattern, kind=kind)
    handle = fs()

    # 先只读 footer 拿到每个文件的行数，用于把 scan_rows 按比例分给各文件。
    sizes: list[tuple[str, int, int, Path]] = []
    for name in names:
        # HfFileSystem 的 open 需要显式 revision，否则会落到 main。
        with handle.open(_remote_path(repo, name, kind), "rb", revision=revision) as remote:
            parquet = pq.ParquetFile(remote)
            sizes.append(
                (name, parquet.metadata.num_rows, parquet.metadata.num_row_groups, Path(name))
            )

    total_rows = sum(item[1] for item in sizes) or 1
    for name, rows, groups, _ in sizes:
        if scan_rows is None:
            wanted = groups
        else:
            share = scan_rows * rows / total_rows
            wanted = max(1, min(groups, math.ceil(share / max(1, rows // max(1, groups)))))
        if wanted >= groups:
            indices = list(range(groups))
        else:
            indices = sorted(
                {round(i * (groups - 1) / max(1, wanted - 1)) for i in range(wanted)}
            )
        with handle.open(_remote_path(repo, name, kind), "rb", revision=revision) as remote:
            parquet = pq.ParquetFile(remote)
            start = 0
            for index in range(groups):
                if index in indices:
                    table = parquet.read_row_group(index, columns=columns)
                    yield name, start, table
                start += parquet.metadata.row_group(index).num_rows


def jsonl_lines(
    repo: str,
    revision: str,
    name: str,
    scan_bytes: int | None = None,
    chunk_bytes: int = 4 << 20,
    kind: str = "dataset",
) -> Iterator[tuple[int, str]]:
    """按字节区间抽样读取远程 JSONL。

    Args:
        repo: 仓库 id。
        revision: 钉死的 commit sha。
        name: 仓库内路径。
        scan_bytes: 期望读取的总字节数；None 表示顺序读完整文件。
        chunk_bytes: 每段读取的字节数。
        kind: `"dataset"` 或 `"model"`。

    Yields:
        `(字节偏移, 一行 JSON 文本)`。

    Notes:
        等距取段再丢弃首尾不完整的行：JSONL 没有索引，只能这样偏均匀地抽样。
        每段起点前的那半行会被跳过，保证不会解析到半截 JSON。
    """
    handle = fs()
    path = _remote_path(repo, name, kind)
    if scan_bytes is None:
        with handle.open(path, "rb", revision=revision) as remote:
            for line_no, line in enumerate(remote):
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    yield line_no, text
        return

    size = handle.info(path)["size"]
    chunks = max(1, math.ceil(scan_bytes / chunk_bytes))
    offsets = sorted(
        {int(round(i * (size - chunk_bytes) / max(1, chunks - 1))) for i in range(chunks)}
    )
    with handle.open(path, "rb", revision=revision) as remote:
        for offset in offsets:
            remote.seek(offset)
            block = remote.read(chunk_bytes)
            if not block:
                continue
            lines = block.split(b"\n")
            # 首行可能是从中间截断的，除非正好从文件头开始。
            if offset != 0:
                lines = lines[1:]
            # 末行同样可能不完整，丢掉。
            lines = lines[:-1]
            for line in lines:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    yield offset, text


def http_json(url: str) -> object:
    """GET 一个返回 JSON 的 URL。

    Args:
        url: 完整地址。

    Returns:
        解析后的 JSON 对象。

    Raises:
        urllib.error.URLError: 网络失败时向上抛出。

    Notes:
        走标准库 urllib，会自动读取 HTTPS_PROXY 环境变量；不需要额外装 requests。
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def http_text(url: str) -> str:
    """GET 一个文本资源。

    Args:
        url: 完整地址。

    Returns:
        UTF-8 解码后的文本。
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")
