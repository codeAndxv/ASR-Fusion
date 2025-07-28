# ASR Fusion

ASR Fusion 是一个支持多种自动语音识别（ASR）模型的Python库，既可以作为独立的API服务器运行，也可以作为SDK包在其他项目中使用。

## 功能特点

- 支持 faster-whisper 和 FunASR 两种主流ASR模型
- 提供与OpenAI API兼容的接口
- 可配置模型类型、大小和运行设备
- 支持文件转录和流式转录
- 既可以作为服务器运行，也可以作为SDK包使用

## 安装

使用uv进行安装：

```bash
uv pip install -e .
```

## 快速开始

### 作为API服务器运行

```bash
python main.py --server --host 0.0.0.0 --port 8000
```

服务器启动后，可以通过以下端点访问：

- `POST /v1/audio/transcriptions` - 音频转录
- `POST /v1/audio/translations` - 音频翻译
- `GET /v1/config` - 获取当前配置
- `POST /v1/config` - 更新配置

### 作为SDK包使用

```python
from asr_fusion import Transcriber

# 初始化转录器
transcriber = Transcriber()

# 转录音频文件
result = transcriber.transcribe_file("audio.wav")
print(result["text"])

# 更换模型
transcriber.set_model("funasr", "small")
result = transcriber.transcribe_file("audio.wav")
print(result["text"])
```

## API接口

### 音频转录

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=small" \
  -F "language=zh"
```

### 音频翻译

```bash
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=small"
```

### 配置管理

获取当前配置：
```bash
curl -X GET "http://localhost:8000/v1/config"
```

更新配置：
```bash
curl -X POST "http://localhost:8000/v1/config" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "funasr", "model_size": "small"}'
```

## 配置文件

项目使用 `config.json` 文件进行配置：

```json
{
    "model_type": "faster-whisper",
    "model_size": "small",
    "device": "cpu",
    "compute_type": "int8"
}
```

配置项说明：
- `model_type`: 模型类型，可选 "faster-whisper" 或 "funasr"
- `model_size`: 模型大小，可选 "tiny", "base", "small", "medium", "large"
- `device`: 运行设备，可选 "cpu" 或 "cuda"
- `compute_type`: 计算类型，可选 "int8", "float16", "float32"

## 命令行工具

```bash
# 作为服务器运行
python main.py --server

# 转录音频文件
python main.py --transcribe audio.wav

# 使用指定模型转录
python main.py --transcribe audio.wav --model-type funasr --model-size small
```

## 依赖

- faster-whisper
- funasr
- fastapi
- uvicorn
- python-multipart

## 许可证

MIT
