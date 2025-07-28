from faster_whisper import WhisperModel
import os
from typing import Optional, Dict, Any

class FasterWhisperModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the faster-whisper model"""
        try:
            model_size = self.config.get("model_size", "small")
            device = self.config.get("device", "cpu")
            compute_type = self.config.get("compute_type", "int8")
            
            # Map model size to actual model names
            model_mapping = {
                "tiny": "tiny",
                "base": "base",
                "small": "small",
                "medium": "medium",
                "large": "large-v3"
            }
            
            actual_model_size = model_mapping.get(model_size, "small")
            
            print(f"Loading faster-whisper model: {actual_model_size} on {device} with {compute_type}")
            self.model = WhisperModel(actual_model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error loading faster-whisper model: {e}")
            raise

    def transcribe_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file"""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        try:
            # Get transcription parameters from config or use defaults
            language = kwargs.get("language", "zh")
            task = kwargs.get("task", "transcribe")  # transcribe or translate
            beam_size = kwargs.get("beam_size", 5)
            temperature = kwargs.get("temperature", 0)
            
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,
                task=task,
                beam_size=beam_size,
                temperature=temperature
            )
            
            # Convert segments to text
            segments_list = []
            for segment in segments:
                segments_list.append({
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                })
            
            result = {
                "text": " ".join([segment.text for segment in segments]),
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration
            }
            
            return result
        except Exception as e:
            print(f"Error transcribing file with faster-whisper: {e}")
            raise

    def transcribe_stream(self, audio_stream, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio stream (not implemented for faster-whisper in this example)"""
        # For simplicity, we'll save the stream to a temporary file and transcribe it
        # In a real implementation, you might want to handle this differently
        raise NotImplementedError("Stream transcription not implemented for faster-whisper in this example")
