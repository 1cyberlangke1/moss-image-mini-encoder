"""数据源注册表。

每个子模块负责一件小事：把某个公开数据源读成 `Record` 流。
清洗规则不放这里（见 `moss_small_encoder.build.POLICIES`），
这样「读取」和「怎么洗」可以分别单测、分别改。
"""

from __future__ import annotations

from . import goku, gustavosta, jtatman, n714, shuixian

SOURCES = {
    "jtatman": jtatman,
    "gustavosta": gustavosta,
    "shuixian": shuixian,
    "goku": goku,
    "n714": n714,
}

__all__ = ["SOURCES"]
