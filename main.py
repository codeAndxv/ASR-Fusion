#!/usr/bin/env python3
"""
Main entry point for ASR Fusion
"""

import sys
import argparse
from asr_fusion.api.server import app
from asr_fusion.models.model_manager import ModelManager
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="ASR Fusion - Audio Speech Recognition")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    model_manager = ModelManager()
    config = model_manager.config.get_server_config()
    
    # Use command line arguments if provided, otherwise use config
    host = args.host or config.get("host", "localhost")
    port = args.port or config.get("port", 8603)
    
    # Run the server
    uvicorn.run(
        "asr_fusion.api.server:app",
        host=host,
        port=port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
