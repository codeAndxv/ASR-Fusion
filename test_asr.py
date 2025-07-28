#!/usr/bin/env python3
"""
Test script for ASR Fusion
"""

import json
import os
from asr_fusion import Transcriber

def test_sdk():
    """Test SDK functionality"""
    print("Testing ASR Fusion SDK...")
    
    # Initialize transcriber
    transcriber = Transcriber()
    
    # Get current config
    config = transcriber.get_config()
    print(f"Current config: {config}")
    
    # Test faster-whisper model
    print("\nTesting faster-whisper model...")
    success = transcriber.set_model("faster-whisper", "tiny")
    if success:
        print("Successfully set model to faster-whisper")
        config = transcriber.get_config()
        print(f"Updated config: {config}")
    else:
        print("Failed to set model to faster-whisper")
    
    # Test funasr model
    print("\nTesting FunASR model...")
    success = transcriber.set_model("funasr", "small")
    if success:
        print("Successfully set model to FunASR")
        config = transcriber.get_config()
        print(f"Updated config: {config}")
    else:
        print("Failed to set model to FunASR")

def test_api_compatibility():
    """Test API compatibility"""
    print("\nTesting API compatibility...")
    
    # This would typically be done with actual API calls
    # For now, we'll just show the expected API structure
    
    # Transcription endpoint
    transcription_request = {
        "file": "audio.wav",
        "model": "small",
        "language": "zh",
        "prompt": "optional prompt",
        "response_format": "json",
        "temperature": 0
    }
    
    print("Transcription request format:")
    print(json.dumps(transcription_request, indent=2, ensure_ascii=False))
    
    # Translation endpoint
    translation_request = {
        "file": "audio.wav",
        "model": "small",
        "prompt": "optional prompt",
        "response_format": "json",
        "temperature": 0
    }
    
    print("\nTranslation request format:")
    print(json.dumps(translation_request, indent=2, ensure_ascii=False))
    
    # Config endpoints
    print("\nConfig endpoints:")
    print("GET /v1/config - Get current configuration")
    print("POST /v1/config - Update configuration")

if __name__ == "__main__":
    test_sdk()
    test_api_compatibility()
    print("\nTest completed!")
