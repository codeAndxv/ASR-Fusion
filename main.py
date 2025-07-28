import argparse
import sys
from asr_fusion.api.server import app
from asr_fusion.sdk.transcriber import Transcriber
import uvicorn
import json

def main():
    parser = argparse.ArgumentParser(description="ASR Fusion - Audio Transcription Tool")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--transcribe", help="Transcribe an audio file")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--model-type", choices=["faster-whisper", "funasr"], help="Model type")
    parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large"], help="Model size")
    
    args = parser.parse_args()
    
    if args.server:
        # Run as API server
        print(f"Starting ASR Fusion API server on {args.host}:{args.port}")
        uvicorn.run("asr_fusion.api.server:app", host=args.host, port=args.port, reload=True)
    elif args.transcribe:
        # Transcribe a file using SDK
        config_file = args.config if args.config else "config.json"
        transcriber = Transcriber(config_file)
        
        # Update model configuration if specified
        if args.model_type:
            model_size = args.model_size if args.model_size else "small"
            transcriber.set_model(args.model_type, model_size)
        
        try:
            result = transcriber.transcribe_file(args.transcribe)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error transcribing file: {e}")
            sys.exit(1)
    else:
        # Show help if no arguments provided
        parser.print_help()

if __name__ == "__main__":
    main()
