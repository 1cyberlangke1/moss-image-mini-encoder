"""把提示词编码成隐状态特征。

- `encoder.py`：`TextEncoder`，教师（量化权重）与学生（目录权重）共用一份实现；
- `base.py`：`_EncoderBase`，补齐与变长两种前向的具体写法；
- `quant.py`：把 safetensors 里的量化权重装成 comfy_kitchen 的量化张量。

都只依赖 pip 包（torch / transformers / comfy_kitchen / safetensors）。
"""

from .encoder import TextEncoder

__all__ = ["TextEncoder"]
