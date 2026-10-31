# 工具脚本 (scripts)

本项目在 `src/scripts/` 中提供模型下载与量化转换的辅助脚本。

---

## 1. 模型批量下载脚本 (`download_models.py`)

用于按预定义分组下载 MOSS-Image 原生底模、教师文本编码器以及 Qwen3.5 学生模型，所有文件均锁定在指定的 Commit SHA；LFS 文件下载后校验 SHA256，非 LFS 文件只校验字节数。

### 常用命令示例

```bash
# 1. 默认下载训练侧所需模型（教师 9B + 学生 0.8B / 2B / 4B + 分词器）
python src/scripts/download_models.py --dest models

# 若本地已有教师模型文件，可指定路径复用以跳过下载
python src/scripts/download_models.py --dest models \
  --teacher-from-local <已有 qwen3.5_9b_w4a8_gptq.safetensors 路径>

# 2. 只下载指定学生模型（例如只下 2B 与 4B，此时不含教师与分词器）
python src/scripts/download_models.py --dest models --only student-2b,student-4b

# 3. 下载 ComfyUI 生成端所需的完整底模套件（DiT int8、VAE、LoRA 及 9B 编码器）
python src/scripts/download_models.py --dest <ComfyUI路径>/models --group comfy

# 4. 查看所有受支持的模型组件注册清单
python src/scripts/download_models.py --list

# 5. 校验指定目录下已下载的文件（LFS 比 SHA256，非 LFS 比字节数；不触发重复下载）
python src/scripts/download_models.py --dest models --check
```

---

## 2. 文本编码器量化脚本 (`quantize_encoder.py`)

将 bf16 全精度的 Qwen3.5 学生模型量化转换为 ComfyUI 可直接读取的 `int8_tensorwise` 或 `asym_w4a8_int8` 紧凑格式，降低显存占用。

### 使用示例

```bash
# 将 0.8B 模型转换为 w4a8 格式
python src/scripts/quantize_encoder.py \
  --src models/qwen3.5-0.8b \
  --out models/qwen3.5-0.8b-w4a8.safetensors \
  --format w4a8

# 将 0.8B 模型转换为 int8 格式
python src/scripts/quantize_encoder.py \
  --src models/qwen3.5-0.8b \
  --out models/qwen3.5-0.8b-int8.safetensors \
  --format int8
```

### 说明
- 量化过程基于 `comfy_kitchen` 的实现，与 ComfyUI 原生运行时的前向算子保持一致；
- 遵循 Qwen 架构特性，自动跳过 `linear_attn.conv1d`、`linear_attn.in_proj_a/b`、`embed_tokens`、`lm_head` 与 `visual` 等关键层。
