# 工具脚本 (scripts)

本项目提供用于权重下载与量化预处理的辅助脚本，不属于 ComfyUI 运行时的核心依赖。

---

## 1. 模型批量下载脚本 (`download_models.py`)

用于按预定义分组下载 MOSS-Image 原生底模、教师文本编码器以及 Qwen3.5 学生模型，所有文件均锁定在指定的 Commit SHA 并进行 SHA256 校验。

### 常用命令示例

```bash
# 1. 默认下载训练侧所需模型（教师 9B + 学生 0.8B + 分词器）
python src/scripts/download_models.py --dest models

# 若已有现成教师模型文件，可指定路径复用以跳过下载
python src/scripts/download_models.py --dest models \
  --teacher-from-local <已有 qwen3.5_9b_w4a8_gptq.safetensors 路径>

# 2. 追加下载特定学生模型（例如 2B 与 4B）
python src/scripts/download_models.py --dest models --only student-2b,student-4b

# 3. 下载 ComfyUI 生成端所需的完整底模套件（DiT int8、VAE、LoRA 及 9B 编码器）
# 直接指定目标路径为 ComfyUI/models
python src/scripts/download_models.py --dest <ComfyUI路径>/models --group comfy

# 4. 查看所有受支持的模型组件注册清单
python src/scripts/download_models.py --list

# 5. 校验指定目录下已下载文件的哈希完整性（不触发重复下载）
python src/scripts/download_models.py --dest models --check
```

---

## 2. 文本编码器量化脚本 (`quantize_encoder.py`)

将 bf16 全精度的 Qwen3.5 学生模型量化转换为 ComfyUI 可直接读取的 `int8_tensorwise` 或 `asym_w4a8_int8` 紧凑格式，显著降低显存占用。

### 使用示例

```bash
# 将 0.8B 模型转换为 w4a8 格式
python src/scripts/quantize_encoder.py \
  --src models/qwen3.5-0.8b \
  --dst models/qwen3.5-0.8b-w4a8.safetensors \
  --format w4a8

# 将 0.8B 模型转换为 int8 格式
python src/scripts/quantize_encoder.py \
  --src models/qwen3.5-0.8b \
  --dst models/qwen3.5-0.8b-int8.safetensors \
  --format int8
```

### 量化实现说明
- 量化过程基于 `comfy_kitchen` 的成熟算法实现，保持与 ComfyUI 运行时前向一致。
- 遵循 Qwen 系列量化规范，自动跳过 `linear_attn.conv1d`、`linear_attn.in_proj_a/b`、`embed_tokens`、`lm_head` 与 `visual` 等敏感层。
