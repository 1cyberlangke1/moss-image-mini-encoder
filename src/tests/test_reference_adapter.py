"""导出格式契约：钉住「从参考项目实测出来的」格式，不依赖任何外部文件喵。

原先这里读 `tmp/refs/` 下一份 21 MB 的参考适配器做对拍，离线就整组跳过——
测试不该依赖外部下载喵。现在改成：用我们自己的 `save_linear` 生成文件，
再用**曾经从真实文件里量出来的契约**去断言喵。

契约来源（实测值，记在这里避免再下载）：
- 张量表：weight / mean_in / std_in / mean_out / std_out（参考项目没有 sink）
- weight 是 F16，其余统计量是 F32
- metadata: format=potatoforge_moss_encoder_projection_v1,
  source_layer=target_layer=last_normalized, alpha=0.01（参考项目）
- data_offsets 连续、从 0 开始、总和等于数据区长度

这里不依赖 safetensors 库：直接按它的头格式解析，装没装都不影响喵。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from moss_small_encoder.train.mlp import save_linear
from moss_small_encoder.train.ridge import RidgeSolution

REQUIRED = {"weight", "mean_in", "std_in", "mean_out", "std_out"}
REFERENCE_FORMAT = "potatoforge_moss_encoder_projection_v1"


def read_header(path: Path) -> tuple[dict, dict, int]:
    """按 safetensors 的头格式读出张量表与 metadata喵。

    输入: path safetensors 文件
    输出: (tensors, metadata, header_bytes)
          tensors 是 名称 -> {dtype, shape, data_offsets}
    预期: 头是 8 字节小端长度 + 该长度的 JSON
    """
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    metadata = raw.pop("__metadata__", {}) or {}
    return raw, metadata, 8 + size


def make_solution(d_in: int = 8, d_out: int = 16, lam: float = 0.01) -> RidgeSolution:
    """造一组形状正确的 Ridge 解，供导出测试使用喵。

    输入: d_in 输入维度；d_out 输出维度；lam 正则强度
    输出: RidgeSolution
    预期: 统计量取标准化的平凡值，便于断言 dtype 与形状
    """
    rng = np.random.default_rng(0)
    return RidgeSolution(
        W=rng.normal(size=(d_in, d_out)).astype(np.float32),
        mean_in=np.zeros(d_in, dtype=np.float32),
        std_in=np.ones(d_in, dtype=np.float32),
        mean_out=np.zeros(d_out, dtype=np.float32),
        std_out=np.ones(d_out, dtype=np.float32),
        lam=lam,
        n_tokens=1234,
    )


def test_tensor_set_is_exactly_the_reference_five(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    save_linear(path, make_solution())

    tensors, _, _ = read_header(path)

    assert set(tensors) == REQUIRED


def test_sink_is_optional_and_appended(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    save_linear(path, make_solution(), sink=np.zeros(16, dtype=np.float32))

    tensors, metadata, _ = read_header(path)

    assert set(tensors) == REQUIRED | {"sink"}
    assert tensors["sink"]["dtype"] == "F32"
    assert tensors["sink"]["shape"] == [16]
    assert metadata["sink_included"] == "true"


def test_sink_length_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_linear(tmp_path / "a.safetensors", make_solution(), sink=np.zeros(15))


def test_shapes_and_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    save_linear(path, make_solution(d_in=8, d_out=16))

    tensors, _, _ = read_header(path)

    assert tensors["weight"]["dtype"] == "F16"
    assert tensors["weight"]["shape"] == [8, 16]
    for name in ("mean_in", "std_in", "mean_out", "std_out"):
        assert tensors[name]["dtype"] == "F32"
    assert tensors["mean_in"]["shape"] == [8]
    assert tensors["mean_out"]["shape"] == [16]


def test_metadata_contract(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    save_linear(path, make_solution(lam=0.01), meta={"val_cosine": 0.85})

    _, metadata, _ = read_header(path)

    assert metadata["format"] == REFERENCE_FORMAT
    assert metadata["source_layer"] == "last_normalized"
    assert metadata["target_layer"] == "last_normalized"
    assert float(metadata["alpha"]) == pytest.approx(0.01)
    assert float(metadata["val_cosine"]) == pytest.approx(0.85)


def test_offsets_are_contiguous_and_cover_the_data_region(tmp_path: Path) -> None:
    path = tmp_path / "a.safetensors"
    save_linear(path, make_solution(), sink=np.zeros(16, dtype=np.float32))

    tensors, _, header_bytes = read_header(path)

    spans = sorted(t["data_offsets"] for t in tensors.values())
    assert spans[0][0] == 0
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end == next_start
    assert spans[-1][1] == path.stat().st_size - header_bytes
