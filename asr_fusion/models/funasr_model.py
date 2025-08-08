
from typing import Dict, Any, Generator
import os
from funasr import AutoModel

class FunASRModel:
    def __init__(self, model_name: str, model_path: str = ".", device: str = "cpu"):
        """
        Initialize FunASR model
        
        Args:
            model_name: Name of the model
            model_path: Path to the model directory
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        
        # Initialize the model
        full_model_path = os.path.join(model_path, model_name) if model_path else model_name
        self.model = AutoModel(model=full_model_path, device=device)
    
    def transcribe_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file
        
        Args:
            audio_file_path: Path to the audio file
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dictionary with transcription result
        """
        # Perform transcription
        result = self.model.generate(input=audio_file_path, **kwargs)
        
        # Convert result to the desired format
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Extract segments and format them
        segments_list = []
        if "sentence_info" in result:
            for i, sentence in enumerate(result["sentence_info"]):
                segment_dict = {
                    "id": i,
                    "seek": 0,
                    "start": sentence.get("start", 0),
                    "end": sentence.get("end", 0),
                    "text": sentence.get("text", ""),
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0
                }
                
                # Add word-level timestamps if available
                if "word_list" in sentence:
                    words_list = []
                    for word_info in sentence["word_list"]:
                        words_list.append({
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                            "word": word_info.get("word", ""),
                            "probability": word_info.get("prob", 0.0)
                        })
                    segment_dict["words"] = words_list
                    
                segments_list.append(segment_dict)
        
        return {
            "task": "transcribe",
            "language": result.get("lang", "zh"),
            "duration": result.get("duration", 0),
            "segments": segments_list
        }
    
    def transcribe_stream(self, audio_chunks: Generator[bytes, None, None], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio stream (stub implementation)
        
        Args:
            audio_chunks: Generator yielding audio chunks
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dictionary with transcription result
        """
        # This is a simplified implementation
        # In a real implementation, you would need to handle streaming properly
        raise NotImplementedError("Streaming transcription not yet implemented for FunASR")
