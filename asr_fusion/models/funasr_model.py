from funasr import AutoModel
import os
import tempfile
import numpy as np
from typing import Optional, Dict, Any

class FunASRModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the FunASR model"""
        try:
            model_size = self.config.get("model_size", "small")
            
            # Map model size to actual model names for FunASR
            model_mapping = {
                "tiny": "damo/speech_campplus_sv_zh-cn_16k-common",
                "base": "damo/speech_campplus_sv_zh-cn_16k-common",
                "small": "damo/speech_campplus_sv_zh-cn_16k-common",
                "medium": "damo/speech_campplus_sv_zh-cn_16k-common",
                "large": "damo/speech_campplus_sv_zh-cn_16k-common"
            }
            
            # For transcription, we'll use a general ASR model
            model_name = "paraformer-zh"  # Default Chinese ASR model
            
            print(f"Loading FunASR model: {model_name}")
            self.model = AutoModel(model=model_name)
        except Exception as e:
            print(f"Error loading FunASR model: {e}")
            raise

    def transcribe_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file"""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        try:
            # Get transcription parameters
            language = kwargs.get("language", "zh")
            
            # Process the audio file
            res = self.model.generate(input=audio_file_path, 
                                    batch_size_s=300,
                                    hotword=kwargs.get("hotword", None))
            
            # Format the result to match OpenAI's API format
            if isinstance(res, list) and len(res) > 0:
                result_text = res[0].get("text", "")
                
                result = {
                    "text": result_text,
                    "segments": [],  # FunASR doesn't provide detailed segments by default
                    "language": language
                }
                
                return result
            else:
                raise RuntimeError("Failed to get transcription result from FunASR")
                
        except Exception as e:
            print(f"Error transcribing file with FunASR: {e}")
            raise

    def transcribe_stream(self, audio_stream, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio stream"""
        try:
            # For stream transcription, we need to handle the audio data
            # This is a simplified implementation - in practice, you might want to
            # handle streaming data differently
            
            # If audio_stream is bytes, save to temporary file
            if isinstance(audio_stream, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_stream)
                    tmp_file_path = tmp_file.name
                
                # Transcribe the temporary file
                result = self.transcribe_file(tmp_file_path, **kwargs)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                return result
            else:
                raise NotImplementedError("Stream transcription expects bytes data")
                
        except Exception as e:
            print(f"Error transcribing stream with FunASR: {e}")
            raise
