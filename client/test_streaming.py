#!/usr/bin/env python3
"""
Test script for ASR Fusion streaming functionality
"""

import sys
import os
import time
# Add parent directory to path to import asr_fusion
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.transcriber import ASRFusionClient

def main():
    # Initialize client
    client = ASRFusionClient(base_url="http://localhost:8604")
    
    # Audio file to transcribe
    audio_file = "/Users/dudu/Files/Project/github/codeAndxv/ASR-Fusion/record_sys_250725_143258.m4a"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        print("Please make sure the audio file exists and the path is correct")
        sys.exit(1)
    
    print("Testing ASR Fusion streaming functionality")
    print(f"Audio file: {audio_file}")
    
    try:
        print("\nTesting streaming transcription:")
        # Transcribe file using streaming
        stream_result = client.transcribe_file(
            file_url=audio_file,
            model="faster-whisper/small",
            stream=True
        )
        
        # Process streaming results
        full_text = ""
        start_time = time.time()
        
        for i, result in enumerate(stream_result):
            if i == 0:
                print("Started receiving stream...")
            
            # Print the type of result
            if "type" in result:
                print(f"Received {result['type']}")
                
                if result["type"] == "transcript.text.delta":
                    # Print the delta text as it arrives
                    delta_text = result.get("delta", "")
                    print(f"  Delta: {delta_text}")
                    full_text += delta_text
                    
                elif result["type"] == "transcript.text.done":
                    # Print final result
                    print(f"  Final text: {result.get('text', '')}")
                    print(f"  Language: {result.get('language', 'unknown')}")
                    print(f"  Duration: {result.get('duration', 0):.2f} seconds")
            
            # Print raw data if it's not JSON
            elif "raw" in result:
                print(f"Raw data: {result['raw']}")
        
        end_time = time.time()
        print(f"\nStreaming transcription completed in {end_time - start_time:.2f} seconds")
        print(f"Full text: {full_text}")
            
    except Exception as e:
        print(f"Error during streaming transcription: {e}")
        print("Make sure the ASR Fusion server is running on http://localhost:8603")
        sys.exit(1)

if __name__ == "__main__":
    main()
