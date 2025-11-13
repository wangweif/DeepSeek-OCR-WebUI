# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DeepSeek-OCR-WebUI 是一个基于 DeepSeek-OCR 模型的智能图像识别 Web 应用程序，提供直观的用户界面和强大的识别能力。该项目支持多种部署方式和平台后端。

## 技术栈

### 核心技术
- **后端框架**: FastAPI + Uvicorn
- **OCR引擎**: DeepSeek-OCR 基于 Transformers
- **图像处理**: Pillow (PIL), PyMuPDF (PDF处理)
- **前端**: 纯 HTML/CSS/JavaScript (单页面应用)
- **容器化**: Docker + Docker Compose

### 多平台支持
- **Apple Silicon**: 原生 MPS (Metal Performance Shaders) 后端
- **NVIDIA GPU**: CUDA 加速后端
- **通用 CPU**: 基础推理后端
- **Docker**: 统一容器化部署

### 关键依赖
- `transformers==4.46.3` - 模型推理引擎
- `tokenizers==0.20.3` - 文本分词器
- `PyMuPDF` - PDF 文件处理
- `fastapi==0.119.1` - Web API 框架
- `uvicorn[standard]==0.38.0` - ASGI 服务器
- `torch` - PyTorch 深度学习框架
- `einops`, `addict`, `easydict` - 模型依赖

## 项目结构

```
DeepSeek-OCR-WebUI/
├── backends/                    # 多平台后端实现
│   ├── mps_backend.py          # Apple Silicon MPS 后端
│   ├── cuda_backend.py         # NVIDIA CUDA 后端
│   ├── cpu_backend.py          # 通用 CPU 后端
│   └── transformers_backend.py # Transformers 统一接口
├── web_service_unified.py      # 统一多平台 Web 服务
├── web_service.py              # 主要 Web 服务 (Docker 使用)
├── ocr_ui_modern.html          # 主要前端界面
├── start.sh                    # 通用启动脚本
├── verify_mac_env.sh           # Mac 环境验证脚本
├── requirements.txt            # 基础依赖 (Linux/Docker)
├── requirements-mac.txt        # Mac 专用依赖
├── docker-compose.yml          # Docker 部署配置
├── Dockerfile                  # Docker 镜像构建
├── DeepSeek-OCR-master/        # OCR 模型源代码
└── assets/                     # UI 资源文件
```

## 开发和部署命令

### 通用开发流程

```bash
# 1. 克隆仓库
git clone https://github.com/neosun100/DeepSeek-OCR-WebUI.git
cd DeepSeek-OCR-WebUI

# 2. 安装依赖
# Linux/Docker:
pip install -r requirements.txt

# Mac (Apple Silicon):
conda create -n deepseek-ocr-mlx python=3.11
conda activate deepseek-ocr-mlx
pip install -r requirements-mac.txt

# 3. 启动服务
./start.sh                      # 自动检测平台并启动
# 或直接运行:
python web_service_unified.py   # 手动启动
python web_service.py          # Docker 版本
```

### 平台特定命令

#### Mac (Apple Silicon)
```bash
# 环境验证
./verify_mac_env.sh

# 创建 conda 环境
conda create -n deepseek-ocr-mlx python=3.11
conda activate deepseek-ocr-mlx

# 安装 PyTorch (MPS 支持)
pip install torch torchvision

# 启动服务
FORCE_BACKEND=mps ./start.sh
```

#### Linux (NVIDIA GPU)
```bash
# 安装 CUDA 版本 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 启动服务
FORCE_BACKEND=cuda ./start.sh
```

#### Docker 部署
```bash
# 构建并启动
docker compose up -d

# 查看日志
docker logs -f deepseek-ocr-webui

# 健康检查
curl http://localhost:8001/health

# 停止服务
docker compose down
```

### 测试和验证命令

```bash
# 健康检查
curl http://localhost:8001/health

# 预期响应:
# {
#   "status": "healthy",
#   "backend": "mps",  # 或 "cuda"/"cpu"
#   "platform": "Darwin",  # 或 "Linux"
#   "model_loaded": true
# }

# 环境验证 (Mac)
./verify_mac_env.sh

# 检查容器状态 (Docker)
docker compose ps
```

## 核心功能特性

### 7 种识别模式
1. **Doc to Markdown** - 文档转 Markdown，保留格式
2. **General OCR** - 通用 OCR 文字提取
3. **Plain Text** - 纯文本提取
4. **Chart Parser** - 图表和公式识别
5. **Image Description** - 图像描述生成
6. **Find & Locate** - 查找并定位特定文本
7. **Custom Prompt** - 自定义识别提示

### 关键特性
- **PDF 支持**: 自动将 PDF 页面转换为图像进行处理
- **批量处理**: 支持多图像顺序识别
- **边界框可视化**: Find 模式自动标注位置
- **多语言支持**: 简体中文、繁体中文、英文、日文
- **ModelScope 回退**: 当 HuggingFace 不可用时自动切换

## 后端架构

### 统一后端接口
项目使用策略模式实现多平台后端:

- `MPSBackend`: Apple Silicon Metal 性能着色器加速
- `CUDABackend`: NVIDIA CUDA GPU 加速
- `CPUBackend`: 通用 CPU 推理
- `TransformersBackend`: 统一的 Transformers 模型接口

### 自动平台检测
`detect_platform()` 函数自动识别运行环境:
- 检查 `FORCE_BACKEND` 环境变量强制指定后端
- Apple Silicon (arm64) + MPS 可用 → MPS 后端
- NVIDIA GPU + CUDA 可用 → CUDA 后端
- 其他 → CPU 后端

### 模型加载和缓存
- 支持从 HuggingFace 和 ModelScope 加载模型
- 自动下载并缓存 ~7GB 模型文件
- 智能重试和超时处理 (5分钟超时)

## 配置和环境变量

### 关键环境变量
```bash
# 服务配置
API_HOST=0.0.0.0              # 监听地址
MODEL_NAME=deepseek-ai/DeepSeek-OCR  # 模型名称
CUDA_VISIBLE_DEVICES=0        # GPU 设备

# 后端强制指定
FORCE_BACKEND=mps             # 强制 MPS 后端
FORCE_BACKEND=cuda            # 强制 CUDA 后端
FORCE_BACKEND=cpu             # 强制 CPU 后端

# Docker 配置
VLLM_USE_V1=0                # vLLM 配置 (已弃用)
PYTHONUNBUFFERED=1           # Python 输出无缓冲
```

### Docker 配置要点
- 基于 `nvcr.io/nvidia/pytorch:25.09-py3` 镜像
- GPU 资源预留和共享内存配置 (`shm_size: "8g"`)
- 模型缓存卷挂载 (`./models:/root/.cache/huggingface`)
- 健康检查配置 (30秒间隔，5分钟启动时间)

## 开发注意事项

### Mac 开发要点
- **必须使用 conda 环境**避免依赖冲突
- 确保安装支持 MPS 的 PyTorch 版本
- 使用 `verify_mac_env.sh` 验证环境配置

### 模型管理
- 首次运行会自动下载模型，需要稳定网络连接
- 模型文件缓存位置: `~/.cache/huggingface` 或 Docker 卷
- ModelScope 作为 HuggingFace 的回退源

### 性能优化
- Apple Silicon: 使用原生 MPS 后端获得最佳性能
- NVIDIA GPU: 确保 CUDA 驱动和 PyTorch CUDA 版本匹配
- 批量处理时注意内存使用情况

### 多语言支持
- 前端使用 `i18n.js` 实现国际化
- 支持语言: 简体中文、繁体中文、英文、日文
- 语言设置自动保存到本地存储

## 常见问题解决

### 模型加载问题
- 检查网络连接和防火墙设置
- 验证 HuggingFace/ModelScope 访问权限
- 确保 20GB+ 可用磁盘空间

### 后端检测问题
- Mac: 检查 PyTorch MPS 支持和 Apple Silicon 兼容性
- Linux: 验证 NVIDIA 驱动和 CUDA 安装
- 所有平台: 确认 Python 版本为 3.11+

### Docker 部署问题
- 确保 NVIDIA Docker 运行时已安装
- 检查 GPU 设备可见性 (`nvidia-smi`)
- 验证共享内存配置充足

## API 端点

主要 API 端点由 FastAPI 自动生成:
- `POST /ocr` - OCR 识别处理
- `GET /health` - 健康检查
- `GET /` - 主页 (返回 HTML 界面)
- 文件上传和批量处理端点

完整 API 文档可通过 `http://localhost:8001/docs` 访问 (FastAPI 自动生成)。