"""线性投影与特征存储的单测（纯 numpy，不需要 torch）。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moss_small_encoder.train import ridge, store


class TestRidgeAccumulator:
    @staticmethod
    def _data(n=4000, d_in=16, d_out=8, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, d_in)).astype(np.float32)
        A = rng.normal(size=(d_in, d_out)).astype(np.float32)
        y = x @ A + 0.5
        return x, y, A

    def test_recovers_linear_map(self):
        x, y, A = self._data()
        acc = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=0)
        acc.add(x, y)
        solution = acc.solve(lam=1e-8)
        pred = solution.predict(x)
        assert ridge.cosine(pred, y) > 0.999
        assert ridge.relative_l2(pred, y) < 0.01
        assert np.allclose(solution.W, A, atol=0.05) or ridge.cosine(
            solution.W, A
        ) > 0.99

    def test_larger_lambda_shrinks(self):
        x, y, _ = self._data()
        acc = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=0)
        acc.add(x, y)
        small = acc.solve(lam=1e1)
        large = acc.solve(lam=1e6)
        assert np.linalg.norm(large.W) < np.linalg.norm(small.W)

    def test_drop_first_ignores_sink(self):
        x, y, _ = self._data(n=2000)
        # 每条「prompt」10 个 token，第 0 个是 attention sink：
        # 方向恒定、幅度极大、目标值被写坏。
        sink = np.arange(0, 2000, 10)
        x = x.copy()
        y = y.copy()
        x[sink] = 1e4
        y[sink] = -5e3
        clean = np.ones(2000, dtype=bool)
        clean[sink] = False

        def feed(acc):
            # drop_first 按调用生效，所以必须按 prompt 分块喂，不能一次喂完。
            for start in range(0, 2000, 10):
                acc.add(x[start : start + 10], y[start : start + 10])
            return acc

        acc_no_drop = feed(ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=0))
        acc_drop = feed(ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=1))

        assert acc_no_drop.n == 2000
        assert acc_drop.n == 2000 - len(sink)
        # 真正的判据：在被污染的数据上，丢掉 sink 的那份在干净 token 上更准。
        cos_drop = ridge.cosine(acc_drop.solve(1e2).predict(x[clean]), y[clean])
        cos_no_drop = ridge.cosine(acc_no_drop.solve(1e2).predict(x[clean]), y[clean])
        assert cos_drop > cos_no_drop

    def test_drop_first_with_merged_lengths(self):
        """拼接传入时，每段各丢前 drop_first 行，与逐 prompt 调用等价。

        这是最容易静默出错的点：合并成大块 GEMM 后，如果误把整块当成
        一条 prompt（只丢第 0 行），sink 会漏进拟合，且没有任何报错。
        """
        x, y, _ = self._data(n=1200)
        seg_len = 10
        lengths = [seg_len] * (1200 // seg_len)
        sink_rows = np.arange(0, 1200, seg_len)

        # 逐段喂（旧调用方式）。
        per_prompt = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=1)
        for start in range(0, 1200, seg_len):
            per_prompt.add(x[start : start + seg_len], y[start : start + seg_len])
        # 拼接喂（新调用方式，一次大 GEMM）。
        merged = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=1)
        merged.add(x, y, lengths=lengths)

        assert per_prompt.n == 1200 - len(sink_rows)
        assert merged.n == per_prompt.n
        # 两种喂法必须给出同一个解。
        assert np.allclose(
            merged.solve(1e2).W, per_prompt.solve(1e2).W, atol=1e-6
        )
        # 反例：不给 lengths 时只丢第 0 行，n 必然不同——防止有人把
        # 「拼接但忘了 lengths」当成等价用法。
        wrong = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=1)
        wrong.add(x, y)
        assert wrong.n == 1200 - 1

    def test_standardization_moments(self):
        rng = np.random.default_rng(3)
        x = (rng.normal(size=(500, 4)) * 3 + 7).astype(np.float32)
        y = rng.normal(size=(500, 2)).astype(np.float32)
        acc = ridge.RidgeAccumulator(4, 2, drop_first=0)
        acc.add(x, y)
        solution = acc.solve(1e2)
        assert np.allclose(solution.mean_in, x.mean(0), atol=1e-4)
        assert np.allclose(solution.std_in, x.std(0), atol=1e-3)

    def test_raises_without_data(self):
        acc = ridge.RidgeAccumulator(4, 2)
        with pytest.raises(ValueError):
            acc.solve(1.0)

    def test_accumulates_incrementally_equals_oneshot(self):
        x, y, _ = self._data(n=1200)
        whole = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=0)
        whole.add(x, y)
        pieces = ridge.RidgeAccumulator(x.shape[1], y.shape[1], drop_first=0)
        for start in range(0, len(x), 300):
            pieces.add(x[start : start + 300], y[start : start + 300])
        assert pieces.n == whole.n
        assert np.allclose(
            pieces.solve(1e2).W, whole.solve(1e2).W, atol=1e-6
        )


class TestMetrics:
    def test_cosine_perfect(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert ridge.cosine(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert ridge.cosine(a, b) == pytest.approx(0.0)

    def test_cosine_ignores_zero_rows(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert ridge.cosine(a, b) == pytest.approx(1.0)

    def test_relative_l2(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[2.0, 0.0]])
        assert ridge.relative_l2(a, b) == pytest.approx(0.5)


class TestFeatureStore:
    def _write(self, tmp_path, n_prompts=5, length=4, d_in=6, d_out=3):
        rng = np.random.default_rng(0)
        lengths = [length] * n_prompts
        with store.FeatureWriter(tmp_path, d_in, d_out) as writer:
            for i in range(n_prompts):
                writer.add(
                    f"p{i}",
                    rng.normal(size=(length, d_in)).astype(np.float32),
                    rng.normal(size=(length, d_out)).astype(np.float32),
                )
        return lengths

    def test_roundtrip(self, tmp_path):
        lengths = self._write(tmp_path)
        loaded = store.FeatureStore.open(tmp_path)
        assert loaded.n_prompts == len(lengths)
        assert loaded.n_tokens == sum(lengths)
        assert loaded.student.shape == (sum(lengths), 6)
        assert loaded.teacher.shape == (sum(lengths), 3)
        assert loaded.prompt_ids == [f"p{i}" for i in range(len(lengths))]

    def test_sink_roundtrip(self, tmp_path):
        self._write(tmp_path)
        sink = np.arange(3, dtype=np.float32) * 1.5
        with store.FeatureWriter(tmp_path, 6, 3) as writer:
            for i in range(2):
                writer.add(
                    f"q{i}",
                    np.zeros((2, 6), dtype=np.float32),
                    np.zeros((2, 3), dtype=np.float32),
                )
            writer.add_sink(sink)
        assert np.allclose(np.load(tmp_path / "sink.npy"), sink)
        import json

        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert meta["sink_excluded"] is True

    def test_rejects_length_mismatch(self, tmp_path):
        with pytest.raises(ValueError):
            with store.FeatureWriter(tmp_path, 4, 2) as writer:
                writer.add(
                    "bad",
                    np.zeros((3, 4), dtype=np.float32),
                    np.zeros((2, 2), dtype=np.float32),
                )

    def test_rejects_wrong_dim(self, tmp_path):
        with pytest.raises(ValueError):
            with store.FeatureWriter(tmp_path, 4, 2) as writer:
                writer.add(
                    "bad",
                    np.zeros((3, 5), dtype=np.float32),
                    np.zeros((3, 2), dtype=np.float32),
                )

    def test_groups(self, tmp_path):
        self._write(tmp_path, n_prompts=3, length=2)
        loaded = store.FeatureStore.open(tmp_path)
        assert loaded.groups().tolist() == [0, 0, 1, 1, 2, 2]

    def test_split_is_deterministic_and_covers_all(self, tmp_path):
        self._write(tmp_path, n_prompts=40, length=1)
        loaded = store.FeatureStore.open(tmp_path)
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        first = loaded.split_by_prompt(ratios)
        second = loaded.split_by_prompt(ratios)
        for key in ratios:
            assert first[key].tolist() == second[key].tolist()
        union = sorted(np.concatenate(list(first.values())).tolist())
        assert union == list(range(40))

    def test_token_indices(self, tmp_path):
        self._write(tmp_path, n_prompts=3, length=2)
        loaded = store.FeatureStore.open(tmp_path)
        assert loaded.token_indices(np.array([0, 2])).tolist() == [0, 1, 4, 5]
        assert loaded.token_indices(np.array([], dtype=int)).size == 0


class TestRidgeCache:
    """Ridge 结果缓存的往返与失效规则。

    缓存是纯加速手段：命中要给出与重算完全相同的结果，未命中或读坏要安静地
    退回重算，不能让训练多一种失败方式。
    """

    @staticmethod
    def _accumulator(d_in: int = 8, d_out: int = 4, n: int = 256, seed: int = 3):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, d_in)).astype(np.float32)
        y = x @ rng.normal(size=(d_in, d_out)).astype(np.float32)
        acc = ridge.RidgeAccumulator(d_in, d_out, drop_first=0)
        acc.add(x, y)
        return acc

    def test_accumulator_roundtrip_gives_same_solution(self, tmp_path):
        """累加器存盘再读回，统计量与解都必须逐元素一致。

        输入: 随机特征累加出的累加器。
        输出: 无（断言）。
        预期: 六组统计量与 n 完全相同；同一 λ 下 W 完全相同（不是近似）。
        """
        acc = self._accumulator()
        path = tmp_path / "stats.npz"
        ridge.save_accumulator(path, acc)

        loaded = ridge.load_accumulator(path, drop_first=0)
        assert loaded is not None
        assert loaded.n == acc.n
        assert (loaded.d_in, loaded.d_out) == (acc.d_in, acc.d_out)
        for name in ("xx", "xy", "sx", "sy", "sx2", "sy2"):
            original = getattr(acc, name).detach().cpu().numpy()
            restored = getattr(loaded, name).detach().cpu().numpy()
            assert np.array_equal(original, restored), name

        assert np.array_equal(acc.solve(1e-3).W, loaded.solve(1e-3).W)

    def test_solution_roundtrip(self, tmp_path):
        """解与 λ 扫描记录存盘再读回，数值与顺序都不变。"""
        solution = self._accumulator().solve(1e-2)
        grid = [(1e-2, 0.5, 0.8), (1e-1, 0.4, 0.9)]
        path = tmp_path / "solution.npz"
        ridge.save_solution(path, solution, grid)

        loaded = ridge.load_solution(path)
        assert loaded is not None
        restored, restored_grid = loaded
        assert restored.lam == solution.lam
        assert restored.n_tokens == solution.n_tokens
        for name in ("W", "mean_in", "std_in", "mean_out", "std_out"):
            assert np.array_equal(getattr(restored, name), getattr(solution, name)), name
        assert restored_grid == grid

    def test_missing_cache_returns_none(self, tmp_path):
        """文件不存在时返回 None，调用方据此回退到重算。"""
        assert ridge.load_accumulator(tmp_path / "nope.npz", drop_first=1) is None
        assert ridge.load_solution(tmp_path / "nope.npz") is None

    def test_corrupt_cache_returns_none(self, tmp_path):
        """读坏的文件只降级成重算，不抛异常打断训练。"""
        path = tmp_path / "broken.npz"
        path.write_bytes(b"not a npz at all")
        assert ridge.load_accumulator(path, drop_first=1) is None
        assert ridge.load_solution(path) is None

    def test_stats_key_depends_on_meta_and_drop_first(self):
        """换特征或换 sink 口径都必须换键，避免误命中旧缓存。"""
        base = ridge.ridge_stats_key(b'{"a":1}', drop_first=1)
        assert base == ridge.ridge_stats_key(b'{"a":1}', drop_first=1)
        assert base != ridge.ridge_stats_key(b'{"a":2}', drop_first=1)
        assert base != ridge.ridge_stats_key(b'{"a":1}', drop_first=0)

    def test_solution_key_depends_on_grid_and_eval_tokens(self):
        """换 λ 网格、验证 token 数或上游键都必须换键。"""
        base = ridge.ridge_solution_key("abc", (1e0, 1e1), 50_000)
        assert base == ridge.ridge_solution_key("abc", (1e0, 1e1), 50_000)
        assert base != ridge.ridge_solution_key("abc", (1e0, 1e2), 50_000)
        assert base != ridge.ridge_solution_key("abc", (1e0, 1e1), 1_000)
        assert base != ridge.ridge_solution_key("abd", (1e0, 1e1), 50_000)


class _FakeReader:
    """最小可用的特征读取器替身。

    只实现 accumulate_train 用到的四个属性，用于验证双缓冲预取
    「不丢块、不重块、顺序无关」——结果必须与串行累加逐元素相同。
    """

    def __init__(self, lengths, d_in=6, d_out=4, seed=0):
        self.lengths = list(lengths)
        self.d_in, self.d_out = d_in, d_out
        self.n_tokens = int(sum(self.lengths))
        rng = np.random.default_rng(seed)
        self._x = rng.normal(size=(self.n_tokens, d_in)).astype(np.float16)
        self._y = rng.normal(size=(self.n_tokens, d_out)).astype(np.float16)
        self._starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(int)

    def read_chunk_into(self, indices, x_buf, y_buf, drop_first=0):
        keep = []
        for i in idx_list(indices):
            start = int(self._starts[i])
            end = start + int(self.lengths[i])
            begin = start + min(drop_first, end - start)
            if begin < end:
                keep.append(np.arange(begin, end))
        rows = np.concatenate(keep) if keep else np.array([], dtype=int)
        n = int(rows.size)
        x_buf[:n] = self._x[rows]
        y_buf[:n] = self._y[rows]
        return n


def idx_list(indices):
    return [int(i) for i in indices]


class TestAccumulatePrefetch:
    """双缓冲预取必须与串行累加得到完全相同的统计量。"""

    def test_matches_serial_accumulation(self):
        """块大小小到让两个缓冲槽反复交替，结果仍与逐条串行一致。

        输入: 14 条长度不一的假特征。
        输出: 无（断言）。
        预期: xx / xy / n 与串行累加逐元素相同——预取只允许改变时序，
              不允许改变内容。
        """
        from moss_small_encoder.train.run import accumulate_train

        lengths = [5, 3, 7, 2, 9, 4, 6, 8, 3, 5, 11, 2, 4, 6]
        reader = _FakeReader(lengths)
        prompts = np.arange(len(lengths))
        got = accumulate_train(reader, prompts, chunk_tokens=4, device="cpu")

        want = ridge.RidgeAccumulator(reader.d_in, reader.d_out, drop_first=1, device="cpu")
        x_buf = np.empty((20, reader.d_in), dtype=np.float16)
        y_buf = np.empty((20, reader.d_out), dtype=np.float16)
        for i in range(len(lengths)):
            n = reader.read_chunk_into([i], x_buf, y_buf, drop_first=1)
            want.add(x_buf[:n], y_buf[:n], clean=True)

        assert got.n == want.n
        for name in ("xx", "xy", "sx", "sy", "sx2", "sy2"):
            left = getattr(got, name).detach().cpu()
            right = getattr(want, name).detach().cpu()
            assert torch.allclose(left, right, atol=1e-6), name
