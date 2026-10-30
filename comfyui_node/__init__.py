"""ComfyUI 节点：把预计算好的 MOSS 条件张量送进采样器。

用途: 用 Qwen3.5-0.8B + 线性投影替代 MOSS-Image 的 9B 文本编码器。
条件由项目侧脚本 tmp/make_cond.py 预先算好（那里才能跑 HF 的 Qwen3.5 实现），
本节点只负责读文件并包成 ComfyUI 的 CONDITIONING，因此不需要 transformers。

安装: 把本目录链接或复制到 ComfyUI/custom_nodes/ 下，重启 ComfyUI。
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


class MossCondLoader:
    """读取 [1, T, 4096] 条件张量，输出 CONDITIONING。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "make_cond.py 生成的 .safetensors 绝对路径",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "conditioning/MOSS"
    DESCRIPTION = "Load a precomputed MOSS conditioning tensor produced by a smaller text encoder."

    def load(self, cond_path: str):
        """读取条件文件。

        Args:
            cond_path: `.safetensors` 路径，内含 `cond` 张量，形状 `[T, 4096]` 或 `[1, T, 4096]`。

        Returns:
            `(CONDITIONING,)`，即 `[[cond, {}]]`。

        Raises:
            ValueError: 文件不存在或形状/取值非法。
        """
        path = Path(cond_path.strip().strip('"'))
        if not path.is_file():
            raise ValueError(f"找不到条件文件：{path}")
        data = load_file(str(path))
        if "cond" not in data:
            raise ValueError(f"{path.name} 里没有 'cond' 张量，实际键：{list(data)}")
        cond = data["cond"]
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)
        if cond.ndim != 3 or cond.shape[-1] != 4096:
            raise ValueError(f"cond 形状应为 [1,T,4096]，实际 {tuple(cond.shape)}")
        if not bool(torch.isfinite(cond).all()):
            raise ValueError("cond 含非有限值")
        return ([[cond.float(), {}]],)


NODE_CLASS_MAPPINGS = {"MossCondLoader": MossCondLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MossCondLoader": "MOSS Load Conditioning (0.8B)"}

# 诊断节点：把条件导出成文件，用于跨实现对齐校验
try:
    from .dumper import NODE_CLASS_MAPPINGS as _DUMP_CLS
    from .dumper import NODE_DISPLAY_NAME_MAPPINGS as _DUMP_NAME

    NODE_CLASS_MAPPINGS.update(_DUMP_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_DUMP_NAME)
except Exception as error:  # noqa: BLE001 - 诊断节点缺失不应影响主节点
    print(f"[moss-08b-adapter] dumper 未加载: {error}", flush=True)
