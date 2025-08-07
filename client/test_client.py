#!/usr/bin/env python3
"""
Test client for ASR Fusion API
"""

import argparse
import sys
import os
from asr_fusion import ASRFusionClient

def test_file_transcription(client, file_path, model="faster-whisper/large-v3"):
    """Test file transcription"""
    print(f"Testing file transcription with model: {model}")
    print(f"File: {file_path}")
    
    try:
        result = client.transcribe_file(
            file_path=file_path,
            model=model,
            response_format="json"
        )
        
        print("Transcription result:")
        print(f"Language: {result['language']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print("\nSegments:")
        for segment in result['segments']:
            print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
        
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def test_text_format(client, file_path, model="faster-whisper/large-v3"):
    """Test text format output"""
    print(f"\nTesting text format output with model: {model}")
    
    try:
        result = client.transcribe_file(
            file_path=file_path,
            model=model,
            response_format="text"
        )
        
        print("Transcription result (text format):")
        print(result)
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test ASR Fusion API Client")
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8603",
        help="ASR Fusion API server URL"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Audio file to transcribe"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="faster-whisper/large-v3",
        help="Model to use for transcription"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text"],
        default="json",
        help="Response format"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Initialize client
    client = ASRFusionClient(base_url=args.host)
    
    # Test transcription
    if args.format == "json":
        test_file_transcription(client, args.file, args.model)
    else:
        test_text_format(client, args.file, args.model)

if __name__ == "__main__":
    main()
