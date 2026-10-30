"""残差 MLP 的单测。

没装 torch 时整组跳过：数据集构建不需要 torch。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from moss_small_encoder.train import mlp, ridge, store  # noqa: E402


class TestBuild:
    def test_shapes_depth1(self):
        net = mlp.build(d_in=8, d_out=4, hidden=16, depth=1)
        out = net(torch.zeros(3, 8))
        assert out.shape == (3, 4)

    def test_shapes_depth3(self):
        net = mlp.build(d_in=8, d_out=4, hidden=16, depth=3)
        out = net(torch.zeros(5, 8))
        assert out.shape == (5, 4)

    def test_zero_init_outputs_zero(self):
        net = mlp.build(d_in=8, d_out=4, hidden=16, depth=2)
        out = net(torch.randn(7, 8))
        assert torch.allclose(out, torch.zeros(7, 4))

    def test_nonzero_init_is_not_zero(self):
        net = mlp.build(d_in=8, d_out=4, hidden=16, depth=1, zero_init=False)
        assert not torch.allclose(net(torch.randn(7, 8)), torch.zeros(7, 4))

    def test_rejects_bad_depth(self):
        with pytest.raises(ValueError):
            mlp.build(d_in=8, d_out=4, hidden=16, depth=0)


class TestTrainNeverWorseThanRidge:
    """零初始化残差的意义：训练再差也不该比线性解更差。"""

    @staticmethod
    def _store(tmp_path, n_prompts=60, length=6, d_in=8, d_out=4, seed=0):
        rng = np.random.default_rng(seed)
        with store.FeatureWriter(tmp_path, d_in, d_out) as writer:
            for i in range(n_prompts):
                x = rng.normal(size=(length, d_in)).astype(np.float32)
                # 目标是非线性函数，线性只能拟合一部分，给残差网络留出空间。
                y = (np.tanh(x) @ rng.normal(size=(d_in, d_out))).astype(np.float32)
                writer.add(f"p{i:03d}", x, y)
        return store.FeatureStore.open(tmp_path)

    def _ridge(self, data):
        acc = ridge.RidgeAccumulator(data.d_in, data.d_out, drop_first=0)
        for i in range(data.n_prompts):
            idx = data.token_indices(np.array([i]))
            acc.add(data.student[idx], data.teacher[idx])
        return acc.solve(1e3)

    def test_best_val_not_below_ridge(self, tmp_path):
        data = self._store(tmp_path)
        solution = self._ridge(data)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(hidden=32, depth=1, epochs=3, batch=64, patience=2, eval_tokens=500),
            device="cpu",
            log=lambda *_: None,
        )
        # 第 0 步输出等于 Ridge，所以最好值不可能低于线性基线。
        assert result.best_val_cosine >= result.base_val_cosine - 1e-4

    def test_records_history_and_epochs(self, tmp_path):
        data = self._store(tmp_path)
        solution = self._ridge(data)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(hidden=16, depth=1, epochs=4, batch=64, patience=4, eval_tokens=300),
            device="cpu",
            log=lambda *_: None,
        )
        assert 1 <= result.epochs_run <= 4
        assert len(result.history) == result.epochs_run
        assert all(isinstance(h, tuple) and len(h) == 3 for h in result.history)

    def test_raises_on_empty_split(self, tmp_path):
        data = self._store(tmp_path, n_prompts=2)
        solution = self._ridge(data)
        with pytest.raises(ValueError):
            mlp.train(
                data,
                solution,
                mlp.MLPConfig(hidden=8, depth=1, epochs=1, batch=8),
                device="cpu",
                log=lambda *_: None,
            )


class TestLinearMode:
    """联合训练 vs 只训残差。"""

    @staticmethod
    def _setup(tmp_path, seed=0):
        data = TestTrainNeverWorseThanRidge._store(tmp_path, seed=seed)
        solution = TestTrainNeverWorseThanRidge()._ridge(data)
        return data, solution

    def test_frozen_keeps_ridge_weights_exactly(self, tmp_path):
        data, solution = self._setup(tmp_path)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(
                hidden=16, depth=1, epochs=3, batch=64, patience=3,
                eval_tokens=300, linear_mode="frozen",
            ),
            device="cpu",
            log=lambda *_: None,
        )
        assert np.array_equal(result.W, solution.W)

    def test_joint_updates_weights(self, tmp_path):
        data, solution = self._setup(tmp_path)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(
                hidden=32, depth=1, epochs=6, batch=64, patience=6,
                eval_tokens=300, linear_mode="joint", lr=5e-3,
            ),
            device="cpu",
            log=lambda *_: None,
        )
        assert result.W.shape == solution.W.shape
        # 从 Ridge 解出发后继续训练，W 应该动了，但幅度不该失控。
        delta = np.linalg.norm(result.W - solution.W) / np.linalg.norm(solution.W)
        assert 0.0 < delta < 0.5

    def test_joint_never_below_ridge_on_validation(self, tmp_path):
        data, solution = self._setup(tmp_path)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(
                hidden=32, depth=1, epochs=4, batch=64, patience=4,
                eval_tokens=300, linear_mode="joint",
            ),
            device="cpu",
            log=lambda *_: None,
        )
        assert result.best_val_cosine >= result.base_val_cosine - 1e-4


class TestSkipZero:
    def test_skip_zero_runs_without_ridge(self, tmp_path):
        data = TestTrainNeverWorseThanRidge._store(tmp_path)
        solution = TestTrainNeverWorseThanRidge()._ridge(data)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(
                hidden=16, depth=1, epochs=2, batch=64, patience=2,
                eval_tokens=300, skip="zero",
            ),
            device="cpu",
            log=lambda *_: None,
        )
        # 没有 Ridge 兜底时，允许比线性差，但要能跑完并给出数字。
        assert result.epochs_run >= 1
        assert isinstance(result.best_val_cosine, float)

class TestSave:
    def test_writes_fp16_tensors_and_metadata(self, tmp_path):
        safetensors = pytest.importorskip("safetensors")
        from safetensors import safe_open

        data = TestTrainNeverWorseThanRidge._store(tmp_path / "feat")
        solution = TestTrainNeverWorseThanRidge()._ridge(data)
        result = mlp.train(
            data,
            solution,
            mlp.MLPConfig(hidden=8, depth=1, epochs=1, batch=32, patience=1, eval_tokens=100),
            device="cpu",
            log=lambda *_: None,
        )
        out = tmp_path / "adapter.safetensors"
        mlp.save(out, solution, result, mlp.MLPConfig(hidden=8, depth=1), meta={"tap": 16})
        assert out.is_file()
        with safe_open(str(out), framework="pt") as handle:
            meta = handle.metadata() or {}
            keys = list(handle.keys())
            assert meta["format"] == "moss_encoder_projection_v1"
            assert meta["tap"] == "16"
            assert meta["hidden"] == "8"
            assert meta["source_layer"] == "last_normalized"
            assert meta["target_layer"] == "last_normalized"
            # 精度按参考项目实际发布文件：权重 fp16，统计量 fp32。
            assert handle.get_tensor("weight").dtype == torch.float16
            assert handle.get_tensor("mean_in").dtype == torch.float32
            assert handle.get_tensor("std_out").dtype == torch.float32
            assert any(k.startswith("mlp.") for k in keys)
        assert out.with_suffix(".json").is_file()
