from faster_whisper import WhisperModel
from typing import Dict, Any, Generator, List
import os
import json

class FasterWhisperModel:
    def __init__(self, model_name: str, model_path: str, device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize FasterWhisper model
        
        Args:
            model_name: Name of the model (e.g., "large-v3")
            model_path: Path to the model directory
            device: Device to run the model on ("cpu" or "cuda")
            compute_type: Compute type for the model ("int8", "float16", etc.)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type

        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def transcribe_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file
        
        Args:
            audio_file_path: Path to the audio file
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dictionary with transcription result
        """
        timestamp_granularities = ["segments"]
        if "timestamp_granularities" in kwargs:
            timestamp_granularities = kwargs["timestamp_granularities"]

        print(f"Faster-Whisper start transcribe file: {audio_file_path}")
        segments, transcription_info = self.model.transcribe(audio_file_path, **kwargs)
        
        # Convert segments to the desired format
        segments_list = []
        words_list = []
        for segment in segments:
            segment_dict = {
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
            }

            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words_list.append({
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability
                    })

            segments_list.append(segment_dict)
        full_text = "/n".join(segment.text for segment in segments).strip()

        transcription_result = {
            "task": "transcribe",
            "language": transcription_info.language,
            "duration": transcription_info.duration,
            "text": full_text,
        }

        if "segments" in timestamp_granularities:
            transcription_result["segments"] = segments_list
        if "word" in timestamp_granularities:
            transcription_result["words"] = words_list
        return transcription_result

    def transcribe_file_to_streaming(self, audio_file_path: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Transcribe audio stream and yield results in OpenAI format
        
        Args:
            audio_file_path: Path to the audio file
            **kwargs: Additional arguments for transcription
            
        Yields:
            Dictionary with transcription result in OpenAI format
        """

        print(f"Faster-Whisper start transcribe file: {audio_file_path}")
        segments, transcription_info = self.model.transcribe(audio_file_path, **kwargs)
        
        # Collect all segments and their words
        full_text = ""
        
        for segment in segments:
            full_text = segment.text + "/n"
            yield {
                "type":"transcript.text.delta",
                "delta": segment.text,
            }
        full_text = full_text.strip()

        # Yield final result
        yield {
            "type": "transcript.text.done",
            "language": transcription_info.language,
            "duration": transcription_info.duration,
            "text": full_text,
        }

