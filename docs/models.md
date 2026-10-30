# 模型资源与下载配置

本项目通过 `src/scripts/download_models.py` 脚本统一管理模型下载与完整性校验。资源按用途划分为不同的组件组：

| 组件组 | 包含内容 | 适用场景 |
| --- | --- | --- |
| `train`（默认组） | 教师模型（9B w4a8）、学生模型（Qwen3.5-0.8B / 2B / 4B）及分词器 | 特征提取与投影层训练 |
| `comfy` | DiT int8 底模（SFT/Base）、FLUX.2 VAE、4 步 / 8 步蒸馏 LoRA、9B 文本编码器 | ComfyUI 生成与推理环境部署 |
| `extra` | DiT bf16 全精度底模、9B bf16 / int8 文本编码器 | 精度对照实验与量化基准测试 |

---

## 常用下载命令

```bash
# 查看所有支持的模型条目清单
python src/scripts/download_models.py --list

# 下载训练依赖组件（教师模型 + 0.8B 学生 + 分词器）
python src/scripts/download_models.py --dest models

# 追加下载指定学生模型（如 4B 学生）
python src/scripts/download_models.py --dest models --only student-4b

# 下载 ComfyUI 出图所需完整底模套件（可直接指定至 ComfyUI 根目录下的 models）
python src/scripts/download_models.py --dest <ComfyUI路径>/models --group comfy

# 校验已下载文件的 SHA256 完整性（不触发重复下载）
python src/scripts/download_models.py --dest models --check
```

*各文件下载锁定在固定的 Hugging Face Commit SHA 节点，并在下载完成后自动执行 SHA256 完整性校验。*

---

## 训练依赖模型规格

### 1. 教师模型 (Teacher)
MOSS-Image 官方 9B w4a8 文本编码器：

| 属性 | 配置值 |
| --- | --- |
| 仓库 ID | `Comfy-Org/MOSS-Image` |
| Commit SHA | `530a4c94f92c35ef6818b1b035a983fe9dcf5f54` |
| 仓库路径 | `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors` |
| 文件体积 | `7,343,591,088` 字节 |
| SHA256 | `4cc6a1543ad2061806da46f35f90cadb4801dcba246c44782cf42ec0feb6407c` |
| 存储格式 | `asym_w4a8_int8`（带 Lloyd-Max 码本及通道尺度） |

### 2. 学生模型 (Student)
Qwen3.5 官方基础模型：

| 注册标识 | 仓库 ID | Commit SHA | 目标路径 |
| --- | --- | --- | --- |
| `student-0.8b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | `models/qwen3.5-0.8b/` |
| `student-2b` | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `models/qwen3.5-2b/` |
| `student-4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `models/qwen3.5-4b/` |

- 0.8B 主权重文件为 `model.safetensors-00001-of-00001.safetensors`（1,746,942,600 字节，SHA256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`）。

### 3. 分词器 (Tokenizer)
- 提取自 `student-0.8b` 仓库的 `tokenizer.json` 文件（12,807,982 字节，SHA256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`）。
- Qwen3.5 系列全尺寸模型共用相同词表与 Token 映射。

---

## 推理与生成底模（`--group comfy`）

| 注册标识 | 来源仓库与 Revision | 仓库内相对路径 | 目标文件名 |
| --- | --- | --- | --- |
| `dit-sft-int8` | `Comfy-Org/MOSS-Image` @ `530a4c94` | [`diffusion_models/moss_image_sft_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors) | 同原名 |
| `dit-base-int8` | `Comfy-Org/MOSS-Image` @ `530a4c94` | [`diffusion_models/moss_image_base_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_base_int8_convrot.safetensors) | 同原名 |
| `vae-flux2-small` | `Comfy-Org/MOSS-Image` @ `530a4c94` | [`vae/flux2-vae_small.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae_small.safetensors) | 同原名 |
| `text-encoder-9b` | `Comfy-Org/MOSS-Image` @ `530a4c94` | [`text_encoders/qwen3.5_9b_w4a8_gptq.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors) | `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors` |
| `lora-sft-4step` | `akhaliq/MOSS-Image-SFT-4-steps` @ `2ded95c3` | [`moss_image_sft_4step_v2.0.safetensors`](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/resolve/main/moss_image_sft_4step_v2.0.safetensors) | `loras/MOSS-Image-SFT-4-steps.safetensors` |
| `lora-sft-8step` | `lodestones/MOSS-Image-SFT-8-steps-dmd` @ `f95df83b` | [`lora_latest.safetensors`](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/resolve/main/lora_latest.safetensors) | `loras/moss_sft_8step_dmd.safetensors` |
| `lora-base-8step` | `lodestones/MOSS-Image-Base-8-steps-dmd` @ `da05cb3b` | [`lora_dmd2.safetensors`](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd/resolve/main/lora_dmd2.safetensors) | `loras/moss_base_8step_dmd2.safetensors` |
