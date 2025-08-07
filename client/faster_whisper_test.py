#!/usr/bin/env python3
"""
Test script for ASR Fusion with Faster Whisper
"""

import sys
import os
# Add parent directory to path to import asr_fusion
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_fusion import ASRFusionClient

def main():
    # Initialize client
    client = ASRFusionClient(base_url="http://localhost:8603")
    
    # Audio file to transcribe
    audio_file = "../record_sys_250725_143258.m4a"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        print("Please make sure the audio file exists and the path is correct")
        sys.exit(1)
    
    print("Testing ASR Fusion with Faster Whisper model")
    print(f"Audio file: {audio_file}")
    
    try:
        # Transcribe file using Faster Whisper model
        result = client.transcribe_file(
            file_path=audio_file,
            model="faster-whisper/small",
            response_format="json"
        )
        
        print("\nTranscription result:")
        print(f"Detected language: {result['language']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        
        print("\nSegments:")
        for segment in result['segments']:
            print("[%.2fs -> %.2fs] %s" % (segment['start'], segment['end'], segment['text']))
            
            # Print word-level timestamps if available
            if 'words' in segment:
                for word in segment['words']:
                    print("  [%.2fs -> %.2fs] %s" % (word['start'], word['end'], word['word']))
                    
    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Make sure the ASR Fusion server is running on http://localhost:8603")
        sys.exit(1)

if __name__ == "__main__":
    main()
