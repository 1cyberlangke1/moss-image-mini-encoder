"""ComfyUI 节点：一个 `MossCLIPLoader` 替代官方 CLIPLoader，多一个适配器下拉。

    MossCLIPLoader(0.8B / 2B / 4B, adapter) → CLIP → CLIPTextEncode（官方）→ KSampler

`adapter` 选 `none` 时行为与官方 `CLIPLoader` 完全一致（9B 走这条）。

安装: 把本目录链接或复制到 `ComfyUI/custom_nodes/` 下，重启 ComfyUI。
适配器放到 `ComfyUI/models/moss_encoder_adapters/`。

历史节点（`MossEncoderCondEncode` / `MossCondLoader` / `MossCondDumper` / `ApplyProjection`）
的文件仍在，但不再注册：正常出图只需要 `MossCLIPLoader`。
"""

from __future__ import annotations

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

try:
    from .encode import register_adapter_folder

    _ADAPTER_DIR = register_adapter_folder()
    from .loader import NODE_CLASS_MAPPINGS as _LOADER_CLS
    from .loader import NODE_DISPLAY_NAME_MAPPINGS as _LOADER_NAME

    NODE_CLASS_MAPPINGS.update(_LOADER_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_LOADER_NAME)
    print(f"[moss-08b-adapter] 适配器目录: {_ADAPTER_DIR}", flush=True)
except Exception as error:  # noqa: BLE001 - 加载失败时给一条可读信息，不让整个 ComfyUI 起不来
    print(f"[moss-08b-adapter] MossCLIPLoader 未加载: {error}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
