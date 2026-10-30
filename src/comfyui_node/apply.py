"""把学习到的逐 token 投影挂到一个 CLIP 上。

只提供内部实现 `_ProjectedClip`：包装官方 CLIP，覆写编码入口，在返回前做投影。
对外只有一个节点 `MossCLIPLoader`（见 `loader.py`）。
"""

from __future__ import annotations

from pathlib import Path

import folder_paths
import torch

from .adapter import adapter_on, load_adapter, load_sink, to_comfy_taps

ADAPTER_FOLDER = "moss_encoder_adapters"
NONE = "none"


def _adapter_choices() -> list[str]:
    """下拉框：`none` 加上目录里所有 `.safetensors`。

    Returns:
        选项列表；目录为空时只有 `none`，此时节点等价于直通喵。
    """
    try:
        names = folder_paths.get_filename_list(ADAPTER_FOLDER)
    except Exception:  # noqa: BLE001 - 注册阶段异常不该拖垮 ComfyUI
        names = []
    return [NONE] + [name for name in names if name.endswith(".safetensors")]


def _default_sink(adapter_path: str) -> str:
    """按命名约定找配套的 sink.npy。

    Args:
        adapter_path: 适配器绝对路径。

    Returns:
        sink 路径；同目录下没有同名 `_sink.npy` 时返回空串喵。

    Notes:
        `qwen3.5-0.8b_linear.safetensors` → `qwen3.5-0.8b_sink.npy`喵。
    """
    path = Path(adapter_path)
    stem = path.stem
    for suffix in ("_linear", "_int8_linear", "_w4a8_linear"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidate = path.with_name(f"{stem}_sink.npy")
    return str(candidate) if candidate.is_file() else ""


class _ProjectedClip:
    """包一层 CLIP：编码完顺手做投影，下游看到的仍是 CLIP。"""

    def __init__(self, clip, adapter_path: str, sink_path: str) -> None:
        self._clip = clip
        self._adapter_path = adapter_path
        self._sink_path = sink_path
        loaded = load_adapter(adapter_path)
        self._loaded = loaded
        self._taps = list(loaded["taps"])
        inner = getattr(clip.cond_stage_model, clip.cond_stage_model.clip)
        self._layer_idx = to_comfy_taps(self._taps, int(inner.num_layers))

    def __getattr__(self, name: str):
        return getattr(self._clip, name)

    def _project(self, cond: torch.Tensor) -> torch.Tensor:
        """把 `[B, taps, T, H]` 投影成 `[B, T, 4096]`。

        输入: ComfyUI 编码器在多抽头模式下返回的 4 维张量
        输出: 投影后的条件张量，dtype 与输入一致
        预期: 宽度与适配器匹配；token 0 换回 teacher 的 sink
        """
        if cond.ndim != 4:
            raise ValueError(
                f"期望多抽头输出 [B, taps, T, H]，实际 {tuple(cond.shape)}；"
                "确认 adapter 的 taps 与模型层数匹配"
            )
        batch, n_taps, n_tok, hidden = cond.shape
        feats = cond.permute(0, 2, 1, 3).reshape(batch, n_tok, n_taps * hidden)
        device = feats.device
        feats = feats.float()
        d_in = int(feats.shape[-1])
        if d_in != int(self._loaded["weight"].shape[0]):
            raise ValueError(
                f"抽头拼接后是 {d_in} 维，适配器要 {int(self._loaded['weight'].shape[0])} 维；"
                f"当前抽头 {self._taps}"
            )

        out_batch = []
        for row in feats:
            head = None
            body = row
            if self._sink_path:
                head = load_sink(self._sink_path).to(device).float().unsqueeze(0)
                body = row[1:]
            w = adapter_on(self._loaded, device)
            z = (body - w["mean_in"]) / w["std_in"]
            proj = (z @ w["weight"] * w["std_out"] + w["mean_out"]).half().float()
            if not bool(torch.isfinite(proj).all()):
                raise ValueError("投影结果含 NaN/Inf，适配器或输入特征异常")
            out_batch.append(torch.cat([head, proj], dim=0) if head is not None else proj)
        return torch.stack(out_batch, dim=0).to(cond.dtype)

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        """覆写编码入口；三种返回形态分别投影后再交回 ComfyUI。

        Args:
            tokens: ComfyUI 的 token 字典。
            return_pooled: 是否同时返回 pooled。
            return_dict: True 时返回 dict（含 cond / pooled_output）。

        Returns:
            与底层 `CLIP.encode_from_tokens` 同构，只是 cond 已被投影喵。
        """
        previous = getattr(self._clip, "layer_idx", None)
        self._clip.layer_idx = self._layer_idx
        try:
            encoded = self._clip.encode_from_tokens(
                tokens, return_pooled=return_pooled, return_dict=return_dict
            )
        finally:
            self._clip.layer_idx = previous

        if return_dict:
            out = dict(encoded)
            out["cond"] = self._project(out["cond"])
            return out
        if return_pooled:
            cond, pooled = encoded
            return self._project(cond), pooled
        return self._project(encoded)

    def encode_from_tokens_scheduled(
        self, tokens, unprojected=False, add_dict: dict | None = None, show_pbar=True
    ):
        """新版 ComfyUI 的编码入口；CLIPTextEncode 走这条。

        Args:
            tokens: token 字典。
            unprojected: 由上游决定是否要未投影的 pooled。
            add_dict: 追加到每个条件上的元数据。
            show_pbar: 是否显示进度条。

        Returns:
            `[[cond, meta], ...]`，cond 已投影成 `[B, T, 4096]`喵。
        """
        previous = getattr(self._clip, "layer_idx", None)
        self._clip.layer_idx = self._layer_idx
        try:
            out = self._clip.encode_from_tokens_scheduled(
                tokens,
                unprojected=unprojected,
                add_dict=add_dict or {},
                show_pbar=show_pbar,
            )
        finally:
            self._clip.layer_idx = previous
        return [[self._project(cond), meta] for cond, meta in out]

    def clone(self, *args, **kwargs):
        """深拷贝底层 clip，并用同样的参数重新包一层。"""
        return _ProjectedClip(
            self._clip.clone(*args, **kwargs), self._adapter_path, self._sink_path
        )
