"""在 ComfyUI 的模型目录下注册适配器文件夹。"""

from __future__ import annotations

import os

import folder_paths

ADAPTER_FOLDER = "moss_encoder_adapters"


def register_adapter_folder() -> str:
    """把 `models/moss_encoder_adapters` 注册成可枚举的模型目录。

    Returns:
        文件夹绝对路径。目录不存在时创建；重复调用幂等。

    Notes:
        必须在节点 INPUT_TYPES 求值前调用，否则适配器下拉拿不到文件列表喵。
    """
    directory = os.path.join(folder_paths.models_dir, ADAPTER_FOLDER)
    os.makedirs(directory, exist_ok=True)
    if ADAPTER_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(ADAPTER_FOLDER, directory)
    return directory
