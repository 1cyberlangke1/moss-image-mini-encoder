"""把提示词编码成 teacher / student 特征。

- `teacher.py`：Qwen3.5-9B（w4a8 量化），出 4096 维最后一层隐状态；
- `quant.py`：把 safetensors 里的量化权重装成 comfy_kitchen 的量化张量。

两个模块都只依赖 pip 包（torch / transformers / comfy_kitchen / safetensors）。
"""

from .teacher import TeacherEncoder

__all__ = ["TeacherEncoder"]
