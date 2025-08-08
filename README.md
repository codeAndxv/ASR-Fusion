# ASR Fusion

ASR Fusion is a Python package for audio transcription using multiple ASR (Automatic Speech Recognition) models. It provides a unified API interface that supports various ASR engines including Faster Whisper, FunASR, and SenseVoice.

## Features

- Support for multiple ASR engines (Faster Whisper, FunASR, SenseVoice)
- RESTful API server with OpenAI-compatible endpoints
- File and streaming transcription support
- Multiple output formats (JSON, text, SRT, VTT)
- Easy model switching via model identifiers
- Configuration-based model management

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd asr-fusion

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Start the API Server

```bash
python main.py --host localhost --port 8603
```

Or using uvicorn directly:

```bash
uvicorn asr_fusion.api.server:app --host localhost --port 8603 --reload
```

### 2. Test with Client

```bash
# Test with the provided client
python client/faster_whisper_test.py
```

Or use the general test client:

```bash
python client/test_client.py --file path/to/audio.wav --model faster-whisper/large-v3
```

## API Usage

The API follows the OpenAI speech recognition API format:

### File Transcription

```bash
curl http://localhost:8603/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.wav" \
  -F model="faster-whisper/large-v3"
```

### Streaming Transcription

To enable streaming, add the `stream=true` parameter:

```bash
curl http://localhost:8603/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.wav" \
  -F model="faster-whisper/large-v3" \
  -F stream="true"
```

### Parameters

- `file`: Audio file to transcribe
- `model`: Model identifier (e.g., "faster-whisper/large-v3")
- `language`: Language code (optional)
- `prompt`: Initial prompt for the transcription (optional)
- `response_format`: Output format ("json", "text", "srt", "verbose_json", "vtt")
- `temperature`: Sampling temperature (default: 0.0)
- `timestamp_granularities`: Timestamp granularities (optional)

## Supported Models

- **Faster Whisper**: `faster-whisper/model-name`
- **FunASR**: `funasr/model-name`
- **SenseVoice**: `sensevoice/model-name`

## Configuration

The `config.yaml` file controls server and model settings:

```yaml
server:
  host: localhost
  port: 8603
  model:
    - faster-whisper:
       - large-v3:   
           device: cpu
           compute_type: int8
           path: "."
```

## SDK Usage

```python
from asr_fusion import ASRFusionClient

# Initialize client
client = ASRFusionClient(base_url="http://localhost:8603")

# Transcribe file
result = client.transcribe_file(
    file_path="audio.wav",
    model="faster-whisper/large-v3"
)

print(result)

# Streaming transcription
stream_result = client.transcribe_file(
    file_path="audio.wav",
    model="faster-whisper/large-v3",
    stream=True
)

# Process streaming results
for chunk in stream_result:
    if chunk["type"] == "transcript.text.delta":
        print(f"Partial result: {chunk['delta']}")
    elif chunk["type"] == "transcript.text.done":
        print(f"Final result: {chunk['text']}")
```
```

## Development

### Project Structure

```
asr-fusion/
├── asr_fusion/
│   ├── api/           # API server implementation
│   ├── routers/       # API route definitions
│   ├── models/        # Model implementations
│   ├── config/        # Configuration handling
│   └── sdk/           # Client SDK
├── client/            # Test clients
├── config.yaml        # Configuration file
└── main.py            # Entry point
```

### Adding New Models

1. Create a new model implementation in `asr_fusion/models/`
2. Update `asr_fusion/models/model_manager.py` to support the new model
3. Add configuration to `config.yaml`

## License

MIT License
