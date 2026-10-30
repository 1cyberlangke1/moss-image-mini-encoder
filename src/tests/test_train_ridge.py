"""线性投影与特征存储的单测（纯 numpy，不需要 torch）。"""

from __future__ import annotations

import numpy as np
import pytest

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
