"""与参考项目**实际发布**的适配器做格式对齐检查。

文件来源：
    https://huggingface.co/PotatoForge/moss-image-4b-encoder-adapter
    moss_qwen35_4b_to_9b_1600_train.safetensors
    sha256 df5ed7fa15f4c628ad6cb3539d9cd16d68f198df9cd1e07740e9bd7b162c954b

放在 `tmp/refs/` 下（不入库）。文件缺失时整组跳过。

这里不依赖 safetensors 库：直接按它的头格式解析，连装没装相关包都不影响。
目的是把「导出格式」钉在真实文件上，而不是钉在我对文档的理解上。
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

REFERENCE = Path("tmp/refs/moss_qwen35_4b_to_9b_1600_train.safetensors")
REFERENCE_SHA256 = "df5ed7fa15f4c628ad6cb3539d9cd16d68f198df9cd1e07740e9bd7b162c954b"

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_file(), reason="缺少 tmp/refs 下的参考适配器"
)


def read_header(path: Path) -> tuple[dict, dict]:
    """按 safetensors 的头格式读出张量表与 metadata。

    Args:
        path: safetensors 文件。

    Returns:
        `(tensors, metadata)`。`tensors` 是 `名称 -> {dtype, shape, data_offsets}`。

    Notes:
        头是 8 字节小端长度 + 该长度的 JSON，内容先元数据后张量表。
    """
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    metadata = raw.pop("__metadata__", {}) or {}
    return raw, metadata


class TestReferenceAdapter:
    def test_sha256_matches_pinned_value(self):
        digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        assert digest == REFERENCE_SHA256

    def test_tensor_set_is_exactly_five(self):
        tensors, _ = read_header(REFERENCE)
        # 参考项目不带 sink：已知缺口，我们会在自己的导出里额外加 sink_out。
        assert set(tensors) == {"weight", "mean_in", "std_in", "mean_out", "std_out"}

    def test_shapes_and_dtypes(self):
        tensors, _ = read_header(REFERENCE)
        assert tensors["weight"]["dtype"] == "F16"
        assert tensors["weight"]["shape"] == [2560, 4096]
        for key, dim in (("mean_in", 2560), ("std_in", 2560), ("mean_out", 4096), ("std_out", 4096)):
            assert tensors[key]["dtype"] == "F32"
            assert tensors[key]["shape"] == [dim]

    def test_metadata_contract(self):
        _, meta = read_header(REFERENCE)
        assert meta["format"] == "potatoforge_moss_encoder_projection_v1"
        assert meta["source_layer"] == "last_normalized"
        assert meta["target_layer"] == "last_normalized"
        # 归一化统计用 F32，权重用 F16：这是我们导出精度的依据。
        assert "alpha" in meta

    def test_offsets_are_contiguous(self):
        tensors, _ = read_header(REFERENCE)
        spans = sorted(v["data_offsets"] for v in tensors.values())
        assert spans[0][0] == 0
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert end == start

    def test_alpha_is_smaller_than_the_sweep_we_planned(self):
        """参考项目选中的 alpha 是 0.01，比 ClipProj 的 1e1…1e6 小得多。

        两个项目的归一化约定不同（token 加权 vs 未加权、标准化方式不同），
        所以这里不断言数值可比，只锁定「我们的 λ 扫描必须包含很小的值」这个事实。
        """
        _, meta = read_header(REFERENCE)
        assert float(meta["alpha"]) < 1.0
