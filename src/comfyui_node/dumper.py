"""ComfyUI 节点：把 CONDITIONING 里的张量导出成 safetensors，用于对齐校验。

用途: 让 ComfyUI 用它自己的编码器（例如原生 Qwen3.5-9B 实现）产生条件，
      与项目侧用 HuggingFace transformers 算出的特征逐元素比较，
      确认两边是否在同一空间。这是判断「训练目标是否正确」的唯一手段。

安装: 与 MossCondLoader 同目录，重启 ComfyUI 后出现在 conditioning/MOSS 下。
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file


class MossCondDumper:
    """把输入的 CONDITIONING 张量写到文件，原样透传。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "save_path": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "导出到的 .safetensors 绝对路径"},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "dump"
    CATEGORY = "conditioning/MOSS"
    OUTPUT_NODE = True
    DESCRIPTION = "Dump a conditioning tensor to disk for cross-implementation comparison."

    def dump(self, conditioning, save_path: str):
        """保存条件张量。

        Args:
            conditioning: ComfyUI 的 CONDITIONING，`[[cond, meta], ...]`。
            save_path: 输出路径。

        Returns:
            原样透传的 `(conditioning,)`。

        Raises:
            ValueError: 路径为空或条件为空。
        """
        path = Path(save_path.strip().strip('"'))
        if not str(path):
            raise ValueError("save_path 不能为空")
        if not conditioning:
            raise ValueError("cond 为空")

        tensors = {}
        for index, (cond, _meta) in enumerate(conditioning):
            tensors[f"cond_{index}"] = cond.detach().float().cpu().contiguous()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"count": str(len(tensors))})
        print(f"[MossCondDumper] 已写出 {len(tensors)} 个张量 -> {path}", flush=True)
        return (conditioning,)


NODE_CLASS_MAPPINGS = {"MossCondDumper": MossCondDumper}
NODE_DISPLAY_NAME_MAPPINGS = {"MossCondDumper": "MOSS Dump Conditioning"}
