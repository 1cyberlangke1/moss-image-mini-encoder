"""ComfyUI 自定义节点入口喵。

输入: ComfyUI 以 `custom_nodes/<本仓库目录>` 的方式加载本文件喵。
输出: 模块级 `NODE_CLASS_MAPPINGS` 与 `NODE_DISPLAY_NAME_MAPPINGS` 两个字典喵。
预期行为: 把 `<仓库根>/src` 加进 `sys.path`，从 `comfyui_node` 导入节点表并原样
转发喵；导入失败时给出可读提示并返回空表，不让 ComfyUI 启动失败喵。

节点实现只有一份，位于 `src/comfyui_node/` 喵。把该目录直接链接或复制进
`custom_nodes` 同样可用，两条路径加载的是同一份代码喵。

训练依赖在 `src/requirements.txt` 喵。
"""

from __future__ import annotations

import sys
from pathlib import Path

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

_SRC = Path(__file__).resolve().parent / "src"

try:
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from comfyui_node import (  # noqa: E402 - sys.path 必须先插好
        NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS,
    )
    from comfyui_node import (
        NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS,
    )

    NODE_CLASS_MAPPINGS.update(_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as error:  # noqa: BLE001 - 加载失败不能拖垮 ComfyUI
    print(f"[moss-08b-adapter] 仓库根入口未加载: {error}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
