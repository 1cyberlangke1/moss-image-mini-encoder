"""一个节点替代官方 CLIPLoader：选权重的同时选投影。

    MossCLIPLoader(clip_name, type=moss_image, adapter) → CLIP
                                                            ↓
                                                    CLIPTextEncode（官方节点）

输出对象和官方 `CLIPLoader` 完全同类，下游不认识我们，也不需要认识。
`adapter` 选 `none` 时直接返回官方加载结果。
"""

from __future__ import annotations

import os

import folder_paths
import nodes as comfy_nodes

from .apply import NONE, _ProjectedClip, _adapter_choices

ADAPTER_FOLDER = "moss_encoder_adapters"
# 只暴露 MOSS 这一种类型：节点名就叫 MOSS，选别的是误用
TYPES = ["moss_image"]


class MossCLIPLoader:
    """加载 Qwen3.5 学生权重并可选地装上投影。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "Qwen3.5 学生权重（0.8B / 2B / 4B），9B 请用官方 CLIPLoader"},
                ),
                "type": (TYPES, {"tooltip": "MOSS-Image 只接受 moss_image"}),
                "adapter": (_adapter_choices(), {"tooltip": "选 none 表示不投影，等价于官方用法"}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "model/loaders"
    DESCRIPTION = (
        "Load a Qwen3.5 student as a MOSS text encoder and optionally attach the learned "
        "projection, so a standard CLIP Text Encode returns 4096-dim conditions."
    )

    def load(self, clip_name, type, adapter, device="default"):
        """加载 CLIP 并按需包装投影。

        Args:
            clip_name: `models/text_encoders` 下的文件名。
            type: 固定 `moss_image`。
            adapter: 适配器文件名，或 `none`。
            device: `default` 或 `cpu`。

        Returns:
            `(clip,)`，官方 CLIP 对象或装好投影的包装对象喵。

        Raises:
            ValueError: 适配器文件找不到。
        """
        clip = comfy_nodes.CLIPLoader().load_clip(clip_name, type, device)[0]
        name = adapter.strip()
        if name.lower() == NONE:
            return (clip,)
        resolved = folder_paths.get_full_path(ADAPTER_FOLDER, name) or name
        if not os.path.isfile(resolved):
            raise ValueError(f"找不到适配器：{name}")
        return (_ProjectedClip(clip, resolved),)


NODE_CLASS_MAPPINGS = {"MossCLIPLoader": MossCLIPLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MossCLIPLoader": "Load CLIP (MOSS + projection)"}
