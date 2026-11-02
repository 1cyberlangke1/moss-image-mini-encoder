"""把 Ridge 闭式解导出成 ComfyUI 适配器喵。

输入: `RidgeSolution`、可选的 sink 向量与元数据喵。
输出: `potatoforge_moss_encoder_projection_v1` 格式的 safetensors + 同名 json喵。
预期: 六个张量一次写全，推理侧只下一个文件喵。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def save_linear(
    path: Path, solution, meta: dict | None = None, sink=None
) -> None:
    """把纯线性 Ridge 参数导出成对标原参考项目的 safetensors。

    Args:
        path: 输出路径。
        solution: Ridge 解，需有 `W` / `mean_in` / `std_in` / `mean_out` / `std_out` / `lam`。
        meta: 额外元数据。
        sink: teacher 的 token 0 向量 `[4096]`。给了就一起写进文件，
            推理侧只下一个文件即可，不会出现训推不一致喵。

    Raises:
        ValueError: sink 维度与输出维度不符喵。
    """
    tensors = {
        "weight": torch.from_numpy(solution.W).half().contiguous(),
        "mean_in": torch.from_numpy(solution.mean_in).float().contiguous(),
        "std_in": torch.from_numpy(solution.std_in).float().contiguous(),
        "mean_out": torch.from_numpy(solution.mean_out).float().contiguous(),
        "std_out": torch.from_numpy(solution.std_out).float().contiguous(),
    }
    if sink is not None:
        vector = np.asarray(sink, dtype=np.float32).reshape(-1)
        out_dim = int(tensors["mean_out"].shape[0])
        if vector.size != out_dim:
            raise ValueError(f"sink 长度应为 {out_dim}，实际 {vector.size}")
        tensors["sink"] = torch.from_numpy(vector.copy()).contiguous()

    metadata = {
        "format": "potatoforge_moss_encoder_projection_v1",
        "source_layer": "last_normalized",
        "target_layer": "last_normalized",
        "alpha": str(solution.lam),
        "sink_included": "true" if sink is not None else "false",
    }
    if meta:
        metadata.update({k: str(v) for k, v in meta.items()})

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    Path(path).with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
