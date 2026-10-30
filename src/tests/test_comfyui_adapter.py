"""ComfyUI 适配器读取与抽头换算的单测（不依赖 ComfyUI，纯 torch/safetensors）。"""

from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from comfyui_node import adapter


def _write_adapter(path, *, d_in=6, meta=None, std_zero=False, sink=True):
    """写一个最小可用的适配器文件。

    输入: 输出路径、输入宽度、metadata、是否让 std 为 0、是否打包 sink
    输出: 写入的路径
    预期: 默认 6 张量（含 sink）；`sink=False` 时只有 5 张量，模拟老产物喵
    """
    tensors = {
        "weight": torch.zeros(d_in, 4096, dtype=torch.float16),
        "mean_in": torch.zeros(d_in, dtype=torch.float32),
        "std_in": torch.ones(d_in, dtype=torch.float32),
        "mean_out": torch.zeros(4096, dtype=torch.float32),
        "std_out": torch.ones(4096, dtype=torch.float32),
    }
    if sink:
        tensors["sink"] = torch.zeros(4096, dtype=torch.float32)
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


class TestAdapterCache:
    """适配器按 (路径, mtime, 大小) 缓存：省掉每次执行都读 50 MB 的开销喵。"""

    def test_second_load_reuses_object(self, tmp_path):
        """同样文件再读一次应拿到同一份对象，不再解析权重。

        输入: 同一个适配器文件读两遍
        输出: 两个返回值
        预期: 是同一个对象（说明走了缓存）喵
        """
        path = _write_adapter(tmp_path / "a.safetensors")
        assert adapter.load_adapter(path) is adapter.load_adapter(path)

    def test_changed_file_invalidates_cache(self, tmp_path):
        """文件被替换后必须重新读，不能继续用旧权重。

        输入: 先写 d_in=6，再改成 d_in=7 并把 mtime 往后推
        输出: 第二次读到的形状
        预期: 拿到新对象且宽度是 7喵
        """
        path = _write_adapter(tmp_path / "b.safetensors", d_in=6)
        first = adapter.load_adapter(path)

        _write_adapter(path, d_in=7)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        second = adapter.load_adapter(path)
        assert second is not first
        assert int(second["weight"].shape[0]) == 7


class TestOptionalSink:
    """sink 是可选张量：新产物打包在文件里，老产物没有就跳过 token 0 替换喵。"""

    def test_packed_sink_is_read(self, tmp_path):
        """文件里有 sink 就一并读出来，长度必须是 4096。

        输入: 含 sink 的适配器
        输出: loaded["sink"]
        预期: 形状 [4096]、float32喵
        """
        loaded = adapter.load_adapter(_write_adapter(tmp_path / "a.safetensors"))
        assert tuple(loaded["sink"].shape) == (4096,)
        assert loaded["sink"].dtype == torch.float32

    def test_missing_sink_is_not_an_error(self, tmp_path):
        """老产物只有 5 张量，加载不能报错，只是没有 sink 键喵。"""
        loaded = adapter.load_adapter(
            _write_adapter(tmp_path / "b.safetensors", sink=False)
        )
        assert "sink" not in loaded
        assert loaded["taps"] == [-1]

    def test_rejects_wrong_sink_length(self, tmp_path):
        """sink 长度不对要报错，否则 token 0 会被错位替换喵。"""
        path = tmp_path / "c.safetensors"
        tensors = {
            "weight": torch.zeros(6, 4096, dtype=torch.float16),
            "mean_in": torch.zeros(6),
            "std_in": torch.ones(6),
            "mean_out": torch.zeros(4096),
            "std_out": torch.ones(4096),
            "sink": torch.zeros(4095),
        }
        save_file(tensors, str(path), metadata={"format": adapter.ADAPTER_FORMAT})
        with pytest.raises(ValueError, match="sink 长度应为 4096"):
            adapter.load_adapter(path)


class TestAdapterOn:
    """投影权重跟着 clip 设备走：CPU 零复制、GPU 搬一次后缓存。"""

    def test_cpu_returns_same_object(self, tmp_path):
        loaded = adapter.load_adapter(_write_adapter(tmp_path / "a.safetensors"))
        assert adapter.adapter_on(loaded, torch.device("cpu")) is loaded

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_cuda_moves_and_caches(self, tmp_path):
        loaded = adapter.load_adapter(_write_adapter(tmp_path / "b.safetensors"))
        first = adapter.adapter_on(loaded, torch.device("cuda"))
        assert first["weight"].device.type == "cuda"
        assert first["mean_in"].device.type == "cuda"
        second = adapter.adapter_on(loaded, torch.device("cuda"))
        assert second is first, "第二次应命中缓存，不能重新搬运"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_cpu_and_cuda_projection_agree(self, tmp_path):
        """同一份特征在 CPU 与 GPU 上投影，取 fp16 后应逐元素一致。"""
        d_in = 8
        gen = torch.Generator().manual_seed(0)
        tensors = {
            "weight": (torch.randn(d_in, 4096, generator=gen) * 0.02).half(),
            "mean_in": torch.randn(d_in, generator=gen),
            "std_in": torch.rand(d_in, generator=gen) + 0.5,
            "mean_out": torch.randn(4096, generator=gen),
            "std_out": torch.rand(4096, generator=gen) + 0.5,
        }
        path = tmp_path / "c.safetensors"
        save_file(tensors, str(path), metadata={"format": adapter.ADAPTER_FORMAT})
        loaded = adapter.load_adapter(path)

        feats = torch.randn(12, d_in, generator=gen)

        def project(device: torch.device) -> torch.Tensor:
            w = adapter.adapter_on(loaded, device)
            body = feats.to(device)
            z = (body - w["mean_in"]) / w["std_in"]
            return (z @ w["weight"] * w["std_out"] + w["mean_out"]).half().float()

        cpu_out = project(torch.device("cpu"))
        cuda_out = project(torch.device("cuda")).cpu()
        assert torch.equal(cpu_out, cuda_out), (
            f"fp16 后最大差 {(cpu_out - cuda_out).abs().max():.6g}"
        )
