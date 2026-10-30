"""编码端切批与显存预算的测试。

覆盖点：
- `make_batches` 只改分组、不改顺序、不丢样本
- 长度离散度限制确实压低了补齐放大
- `auto_student_budget` 在显存充足时给出更大的批次，且落在允许区间
"""

from __future__ import annotations

import torch

from moss_small_encoder.encode.run import STUDENT_DIM, auto_student_budget, make_batches


def _rows(lengths: list[int]) -> list[dict]:
    """按给定长度造样本列表（id 递增，便于校验顺序）。

    Args:
        lengths: 每条样本的 token 数。

    Returns:
        与 `load_prompts` 同形状的行列表。
    """
    return [{"id": f"p{i:04d}", "text": "x", "n_tokens": n} for i, n in enumerate(lengths)]


def _padded_tokens(batches: list[list[dict]]) -> int:
    """补齐后的总 token 数（每批按批内最长对齐）。"""
    return sum(len(b) * max(r["n_tokens"] for r in b) for b in batches)


def test_batches_keep_order_and_cover_every_row() -> None:
    """切批不重排、不丢样本。

    输入: 100 条长度递减的样本
    输出: 摊平后的 id 序列与原列表完全一致；条数与 token 总数不变
    预期: 分组策略（含长度离散度）只影响"谁和谁一批"，不影响行序
    """
    rows = _rows([200 - i for i in range(100)])
    batches = make_batches(rows, budget=4096, max_prompts=64)

    flat = [row["id"] for batch in batches for row in batch]
    assert flat == [row["id"] for row in rows]
    assert sum(len(batch) for batch in batches) == len(rows)
    assert all(len(batch) > 0 for batch in batches)


def test_aspect_limit_reduces_padding() -> None:
    """长度离散度限制确实压低补齐放大。

    输入: 同样的样本，分别用「不限制」和「1.6 倍限制」切批
    输出: 限制后的补齐 token 总数更少，且批内最长/最短比不超过 1.6
    预期: 这是 4096 预算下把补齐放大从约 4.5 倍降到约 2 倍的机制
    """
    # 长短交替，强制出现"长批里混进极短样本"的情形
    lengths = [800, 100, 780, 95, 760, 90, 740, 85, 720, 80] * 5
    rows = _rows(lengths)

    loose = make_batches(rows, budget=4096, max_prompts=64, max_aspect=10**9)
    tight = make_batches(rows, budget=4096, max_prompts=64, max_aspect=1.6)

    assert _padded_tokens(tight) < _padded_tokens(loose)
    for batch in tight:
        longest = max(r["n_tokens"] for r in batch)
        shortest = min(r["n_tokens"] for r in batch)
        assert longest <= shortest * 1.6


def test_single_long_row_takes_its_own_batch() -> None:
    """单条超过预算的样本独占一批。

    输入: 一条 5000 token 的样本，预算 4096
    输出: 它自己一批，不被切成两段
    预期: 长样本不能被切断，否则会破坏 prompt 边界
    """
    rows = _rows([5000, 100, 120, 140])
    batches = make_batches(rows, budget=4096, max_prompts=64)

    assert batches[0] == [rows[0]]
    assert sum(len(b) for b in batches) == len(rows)


def test_auto_student_budget_bounds() -> None:
    """自动批次落在允许区间内，且随抽头增多而下降。

    输入: 1 抽头与 6 抽头两种配置
    输出: 值在 [8192, 65536] 之间且是 2 的幂；6 抽头不比 1 抽头大
    预期: 拼接时要多留 len(taps) 份副本，抽头越多能装的 token 越少
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        one = auto_student_budget(( -1,), device)
        six = auto_student_budget((2, 4, 8, 12, 16, -1), device)
        for value in (one, six):
            assert 8192 <= value <= 65536
            assert value & (value - 1) == 0, "应为 2 的幂"
        assert six <= one
    else:
        # CPU 上没有显存信息，函数不应抛异常
        assert auto_student_budget((-1,), device) >= 8192


def test_auto_student_budget_counts_hidden_states(monkeypatch) -> None:
    """批预算必须按 `output_hidden_states` 的 n_layers+1 份缓存来估。

    输入: 可用显存固定 8 GiB；(1024 维, 24 层) 与 (2560 维, 36 层) 两种学生
    输出: 两者的 token 预算
    预期: 宽+深的那档要明显更小。若只按抽头数估，两者会得到同一个值喵
    """
    total = 8 * 1024**3
    # 测试里没有真卡也没关系：函数只用到设备类型与 mem_get_info 的返回值喵
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (total, total))
    taps = (2, 4, 8, 12, 16, -1)
    small = auto_student_budget(taps, torch.device("cuda"), hidden=1024, n_layers=24)
    big = auto_student_budget(taps, torch.device("cuda"), hidden=2560, n_layers=36)

    assert small >= 32768, "0.8B 在 8 GiB 卡上应还有较大批次"
    assert big <= small // 2, "4B 每 token 的激活是 0.8B 的两倍多，预算要降下来"


def test_student_dim_constant() -> None:
    """学生隐维常量与 6 抽头拼接宽度一致。"""
    assert STUDENT_DIM == 1024
    assert STUDENT_DIM * 6 == 6144


class TestStudentShape:
    """学生宽度与层数必须从模型 config 读，不能写死（2B 是 2048 维）。"""

    def test_reads_text_config(self, tmp_path):
        import json

        from moss_small_encoder.encode.run import _student_shape

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {"hidden_size": 2048, "num_hidden_layers": 24},
                }
            ),
            encoding="utf-8",
        )
        assert _student_shape(tmp_path) == (2048, 24)

    def test_reads_flat_config(self, tmp_path):
        import json

        from moss_small_encoder.encode.run import _student_shape

        (tmp_path / "config.json").write_text(
            json.dumps({"hidden_size": 1024, "num_hidden_layers": 24}), encoding="utf-8"
        )
        assert _student_shape(tmp_path) == (1024, 24)
