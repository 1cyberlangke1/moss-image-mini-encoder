"""ComfyUI 适配器读取与抽头换算的单测（不依赖 ComfyUI，纯 torch/safetensors）。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from comfyui_node import adapter


def _write_adapter(path, *, d_in=6, meta=None, std_zero=False):
    """写一个最小可用的适配器文件。"""
    tensors = {
        "weight": torch.zeros(d_in, 4096, dtype=torch.float16),
        "mean_in": torch.zeros(d_in, dtype=torch.float32),
        "std_in": torch.ones(d_in, dtype=torch.float32),
        "mean_out": torch.zeros(4096, dtype=torch.float32),
        "std_out": torch.ones(4096, dtype=torch.float32),
    }
    if std_zero:
        tensors["std_in"] = torch.zeros(d_in, dtype=torch.float32)
    save_file(tensors, str(path), metadata=meta or {"format": adapter.ADAPTER_FORMAT})
    return path


class TestParseTaps:
    def test_parses_ordered_list(self):
        """抽头顺序必须原样保留：它决定特征拼接顺序。"""
        assert adapter.parse_taps("2,4,8,12,16,-1") == [2, 4, 8, 12, 16, -1]

    def test_tolerates_spaces(self):
        assert adapter.parse_taps(" 2, 4 ,16 ") == [2, 4, 16]

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            adapter.parse_taps(" , ")


class TestToComfyTaps:
    def test_negative_maps_to_final_norm(self):
        """transformers 的 -1（过 final norm）在 ComfyUI 里是 num_layers。"""
        assert adapter.to_comfy_taps([-1], 24) == [24]

    def test_second_last_maps_to_last_block_input(self):
        """transformers 的 -2 是倒数第二个 block 的输出，对应 ComfyUI 下标 23。"""
        assert adapter.to_comfy_taps([-2], 24) == [23]

    def test_positive_indices_unchanged(self):
        assert adapter.to_comfy_taps([2, 4, 8, 12, 16, -1], 24) == [2, 4, 8, 12, 16, 24]


class TestLoadAdapter:
    def test_loads_and_promotes_to_float32(self, tmp_path):
        """weight 存的是 fp16，读出来要升 fp32 供 CPU 投影使用。"""
        path = _write_adapter(tmp_path / "a.safetensors", meta={"format": adapter.ADAPTER_FORMAT, "taps": "2,4,-1"})
        loaded = adapter.load_adapter(path)
        assert loaded["weight"].dtype == torch.float32
        assert tuple(loaded["weight"].shape) == (6, 4096)
        assert loaded["taps"] == [2, 4, -1]
        assert loaded["meta"]["format"] == adapter.ADAPTER_FORMAT

    def test_missing_tensor_is_reported(self, tmp_path):
        """缺张量要给可读错误，而不是 KeyError。"""
        path = tmp_path / "bad.safetensors"
        save_file({"weight": torch.zeros(6, 4096, dtype=torch.float16)}, str(path))
        with pytest.raises(ValueError, match="缺少张量"):
            adapter.load_adapter(path)

    def test_rejects_foreign_format(self, tmp_path):
        path = _write_adapter(tmp_path / "b.safetensors", meta={"format": "something_else"})
        with pytest.raises(ValueError, match="只支持"):
            adapter.load_adapter(path)

    def test_rejects_wrong_output_width(self, tmp_path):
        path = tmp_path / "c.safetensors"
        save_file(
            {
                "weight": torch.zeros(6, 4095, dtype=torch.float16),
                "mean_in": torch.zeros(6),
                "std_in": torch.ones(6),
                "mean_out": torch.zeros(4095),
                "std_out": torch.ones(4095),
            },
            str(path),
            metadata={"format": adapter.ADAPTER_FORMAT},
        )
        with pytest.raises(ValueError, match="weight 形状"):
            adapter.load_adapter(path)

    def test_rejects_zero_std(self, tmp_path):
        """std 为 0 会让标准化除零，必须在加载期拦住。"""
        path = _write_adapter(tmp_path / "d.safetensors", std_zero=True)
        with pytest.raises(ValueError, match="必须为正数"):
            adapter.load_adapter(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="找不到适配器"):
            adapter.load_adapter(tmp_path / "nope.safetensors")


class TestLoadSink:
    def test_loads_vector(self, tmp_path):
        path = tmp_path / "sink.npy"
        np.save(str(path), np.arange(4096, dtype=np.float32))
        vector = adapter.load_sink(path)
        assert tuple(vector.shape) == (4096,)
        assert vector.dtype == torch.float32

    def test_rejects_wrong_length(self, tmp_path):
        path = tmp_path / "short.npy"
        np.save(str(path), np.zeros(4095, dtype=np.float32))
        with pytest.raises(ValueError, match="长度应为 4096"):
            adapter.load_sink(path)
