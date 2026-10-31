"""抽头 hook 收集器的单元测试喵。

覆盖点：
- 正数抽头取的是「该层输出」，顺序与 `taps` 一致
- `-1` 取的是前向返回的 `last_hidden_state`，不是最后一层的裸输出
- 同一层被多个抽头引用时都要填上
- 抽头等于层数或越界时直接报错
"""

from __future__ import annotations

import pytest
import torch

from moss_small_encoder.train.finetune import collect_tap_hidden, decoder_layers


class _Layer(torch.nn.Module):
    """把输入加上固定偏置，用来区分不同层喵。"""

    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回加偏置后的张量喵。"""
        return x + self.bias


class _Output:
    """只带 `last_hidden_state` 的最小输出对象喵。"""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


def _build(count: int = 4):
    """造 count 层假模型与一个会顺序穿过它们的 run 喵。"""
    layers = torch.nn.ModuleList([_Layer(float(i + 1)) for i in range(count)])
    holder: dict[str, torch.Tensor] = {}

    def run():
        """顺序穿过所有层，返回末层归一化后的值当 last_hidden_state喵。"""
        x = torch.zeros(1, 2)
        for layer in layers:
            x = layer(x)
        holder["final"] = x
        return _Output(x)

    return layers, run, holder


def test_positive_taps_take_layer_output_in_order() -> None:
    """正数抽头取对应层输出，顺序与 taps 一致。

    输入: 4 层、每层累加 1/2/3/4，taps=(1, 3)
    输出: 两个张量分别是 1.0 与 6.0（累加：1、3、6、10）
    预期: 抽头 t 取第 t-1 层输出，顺序不打乱
    """
    layers, run, _ = _build(4)
    got = collect_tap_hidden(layers, (1, 3), run)
    assert got[0].flatten()[0].item() == pytest.approx(1.0)
    assert got[1].flatten()[0].item() == pytest.approx(6.0)


def test_minus_one_uses_last_hidden_state() -> None:
    """`-1` 取的是模型输出里的 last_hidden_state。

    输入: 4 层累加得到 10.0，模型返回的 last_hidden_state 也是 10.0
    输出: 抽头 -1 等于 10.0
    预期: 不能取最后一层的裸输出，否则会漏掉 final norm
    """
    layers, run, _ = _build(4)
    got = collect_tap_hidden(layers, (-1,), run)
    assert got[0].flatten()[0].item() == pytest.approx(10.0)


def test_same_layer_can_feed_multiple_taps() -> None:
    """同一层被多个抽头引用时都要填上。

    输入: taps=(2, 2)
    输出: 两个张量都等于第 1 层的输出 3.0
    预期: 不因为去重而漏填喵
    """
    layers, run, _ = _build(4)
    got = collect_tap_hidden(layers, (2, 2), run)
    assert got[0].flatten()[0].item() == pytest.approx(3.0)
    assert got[1].flatten()[0].item() == pytest.approx(3.0)


def test_tap_equal_to_layer_count_is_rejected() -> None:
    """抽头等于层数时直接报错。

    输入: 4 层模型，taps=(4,)
    输出: IndexError
    预期: 该位置对应 final norm，必须用 -1 表达，避免静默取错层喵
    """
    layers, run, _ = _build(4)
    with pytest.raises(IndexError):
        collect_tap_hidden(layers, (4,), run)


def test_decoder_layers_unwraps_peft_style_and_bare_model() -> None:
    """`decoder_layers` 兼容裸模型与 PeftModel 两种结构。

    输入: 一个裸对象、一个 `base_model.model.layers` 结构
    输出: 两者都返回同一个 ModuleList
    预期: 包装层多一层嵌套也不影响取值喵
    """
    layers, _, _ = _build(3)

    class _Bare:
        """只有 layers 的假模型喵。"""

    bare = _Bare()
    bare.layers = layers
    assert decoder_layers(bare) is layers

    inner = _Bare()
    inner.layers = layers
    outer = _Bare()
    outer.model = inner
    wrapped = _Bare()
    wrapped.base_model = outer
    assert decoder_layers(wrapped) is layers
