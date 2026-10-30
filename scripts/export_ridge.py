"""快速导出纯线性 Ridge 最优解适配器权重。

输入：features/v1 (teacher.raw, student.raw, meta.json)
输出：artifacts/projection_linear.safetensors, artifacts/projection_linear.json
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

from moss_small_encoder.train.store import StreamReader
from moss_small_encoder.train.ridge import RidgeAccumulator

def main():
    feat_dir = Path("features/v1")
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projection_linear.safetensors"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载特征库: {feat_dir} (设备: {device})", flush=True)
    reader = StreamReader(feat_dir)
    print(f"语料: {reader.n_prompts} 条 / {reader.n_tokens} token", flush=True)

    splits = reader.split_by_prompt({"train": 0.70, "val": 0.15, "test": 0.15})
    train_prompts = np.sort(splits["train"])
    val_prompts = np.sort(splits["val"])

    # 1. GPU 矩阵累加
    accumulator = RidgeAccumulator(reader.d_in, reader.d_out, device=device)
    chunk_tokens = 65_536
    x_buf = np.empty((chunk_tokens + 4096, reader.d_in), dtype=np.float16)
    y_buf = np.empty((chunk_tokens + 4096, reader.d_out), dtype=np.float16)

    p_idx = 0
    t0 = time.time()
    print("开始流式矩阵累加 (450万 token)...", flush=True)
    while p_idx < len(train_prompts):
        block = []
        cur_tok = 0
        while p_idx < len(train_prompts) and (cur_tok == 0 or cur_tok < chunk_tokens):
            pid = int(train_prompts[p_idx])
            block.append(pid)
            cur_tok += int(reader.lengths[pid])
            p_idx += 1
        n_read = reader.read_chunk_into(block, x_buf, y_buf, drop_first=1)
        if n_read > 0:
            accumulator.add(x_buf[:n_read], y_buf[:n_read], clean=True)

    t_acc = time.time() - t0
    print(f"矩阵累加完成: {accumulator.n:,} token，耗时 {t_acc:.1f}s", flush=True)

    # 2. 验证集求解与打分
    eval_target = min(200_000, int(reader.lengths[val_prompts].sum()))
    val_p_list = []
    val_tok = 0
    for pid in val_prompts:
        val_p_list.append(pid)
        val_tok += int(reader.lengths[pid])
        if val_tok >= eval_target:
            break
    v_n = reader.read_chunk_into(val_p_list, x_buf, y_buf, drop_first=1)
    val_x = x_buf[:v_n]
    val_y = y_buf[:v_n]

    lambdas = (1e-1, 1e1, 1e3, 1e4, 1e5, 1e6)
    best_cos = -1.0
    best_lam = 1e6
    best_sol = None
    best_rel = 1.0

    print("扫描正则化系数 λ...", flush=True)
    for lam in lambdas:
        sol = accumulator.solve(lam)
        cos, rel = sol.score(val_x, val_y, chunk=65536, device=device)
        print(f"  λ {lam:8.0e} | cos {cos:.5f} | relL2 {rel:.4f}", flush=True)
        if cos > best_cos:
            best_cos = cos
            best_lam = lam
            best_sol = sol
            best_rel = rel

    print(f"\n最佳 λ: {best_lam:.0e} | 验证集余弦: {best_cos:.5f} | 相对 L2 误差: {best_rel:.4f}", flush=True)

    # 3. 立即导出 safetensors
    tensors = {
        "weight": torch.from_numpy(best_sol.W).half().contiguous(),
        "mean_in": torch.from_numpy(best_sol.mean_in).float().contiguous(),
        "std_in": torch.from_numpy(best_sol.std_in).float().contiguous(),
        "mean_out": torch.from_numpy(best_sol.mean_out).float().contiguous(),
        "std_out": torch.from_numpy(best_sol.std_out).float().contiguous(),
    }
    # 如果有 attention sink 均值向量，一并写入
    sink_path = feat_dir / "sink.npy"
    if sink_path.exists():
        sink_vec = np.load(sink_path)
        tensors["sink_out"] = torch.from_numpy(sink_vec).float().contiguous()

    meta = {
        "format": "potatoforge_moss_encoder_projection_v1",
        "source_layer": "last_normalized",
        "target_layer": "last_normalized",
        "alpha": str(best_lam),
        "val_cosine": f"{best_cos:.6f}",
        "rel_l2": f"{best_rel:.6f}",
        "train_prompts": str(len(train_prompts)),
        "train_tokens": str(accumulator.n),
        "student": "Qwen/Qwen3.5-0.8B",
        "teacher": "Qwen/Qwen3.5-9B (w4a8)",
    }

    save_file(tensors, str(out_path), metadata=meta)
    out_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n【权重与结果导出成功】: {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    # 同时也导出一份作为默认 projection.safetensors
    default_out = out_dir / "projection.safetensors"
    save_file(tensors, str(default_out), metadata=meta)
    default_out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"【默认权重落盘完成】: {default_out}", flush=True)
    reader.close()

if __name__ == "__main__":
    main()
