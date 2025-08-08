#!/usr/bin/env python3
"""
Test script for ASR Fusion with Faster Whisper
"""

import sys
import os
import time
# Add parent directory to path to import asr_fusion
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.transcriber import ASRFusionClient

def test_regular_transcription(client, audio_file):
    """Test regular (non-streaming) transcription"""
    print("\n" + "="*50)
    print("Testing Regular Transcription")
    print("="*50)
    
    try:
        # Transcribe file using Faster Whisper model
        result = client.transcribe_file(
            file_url=audio_file,
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
        print(f"Error during regular transcription: {e}")
        raise

def test_streaming_transcription(client, audio_file):
    """Test streaming transcription"""
    print("\n" + "="*50)
    print("Testing Streaming Transcription")
    print("="*50)
    
    try:
        # Transcribe file using streaming
        stream_result = client.transcribe_file(
            file_url=audio_file,
            model="faster-whisper/small",
            stream=True
        )
        
        # Process streaming results
        full_text = ""
        start_time = time.time()
        
        print("\nStreaming transcription started...")
        for i, result in enumerate(stream_result):
            # Print the type of result
            if "type" in result:
                if result["type"] == "transcript.text.delta":
                    # Print the delta text as it arrives
                    delta_text = result.get("delta", "")
                    print(f"  Delta: {delta_text}")
                    full_text += delta_text
                    
                elif result["type"] == "transcript.text.done":
                    # Print final result
                    print(f"\nFinal result:")
                    print(f"  Text: {result.get('text', '')}")
                    print(f"  Language: {result.get('language', 'unknown')}")
                    print(f"  Duration: {result.get('duration', 0):.2f} seconds")
            
            # Print raw data if it's not JSON
            elif "raw" in result:
                print(f"Raw data: {result['raw']}")
        
        end_time = time.time()
        print(f"\nStreaming transcription completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during streaming transcription: {e}")
        raise

def main():
    # Initialize client
    client = ASRFusionClient(base_url="http://localhost:8603")
    
    # Audio file to transcribe
    audio_file = "/Users/dudu/Files/Project/github/codeAndxv/ASR-Fusion/record_sys_250725_143258.m4a"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        print("Please make sure the audio file exists and the path is correct")
        sys.exit(1)
    
    print("Testing ASR Fusion with Faster Whisper model")
    print(f"Audio file: {audio_file}")
    
    try:
        # Test regular transcription
        test_regular_transcription(client, audio_file)
        
        # Test streaming transcription
        test_streaming_transcription(client, audio_file)
        
        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("="*50)
                    
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure the ASR Fusion server is running on http://localhost:8603")
        sys.exit(1)

if __name__ == "__main__":
    main()
