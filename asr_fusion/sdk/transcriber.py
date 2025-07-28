from typing import Dict, Any, Optional
from asr_fusion.models.model_manager import ModelManager
from asr_fusion.config.config import Config
import os

class Transcriber:
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the Transcriber with a configuration file
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config = Config(config_file)
        self.model_manager = ModelManager(self.config)

    def transcribe_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file
        
        Args:
            audio_file_path (str): Path to the audio file
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        return self.model_manager.transcribe_file(audio_file_path, **kwargs)

    def transcribe_stream(self, audio_stream, **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio stream
        
        Args:
            audio_stream: Audio stream data (bytes)
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        return self.model_manager.transcribe_stream(audio_stream, **kwargs)

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update the configuration
        
        Args:
            new_config (Dict[str, Any]): New configuration values
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.model_manager.update_config(new_config)

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            Dict[str, Any]: Current configuration
        """
        return self.config.get_all()

    def set_model(self, model_type: str, model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> bool:
        """
        Set the model configuration
        
        Args:
            model_type (str): Model type ("faster-whisper" or "funasr")
            model_size (str): Model size ("tiny", "base", "small", "medium", "large")
            device (str): Device to run on ("cpu" or "cuda")
            compute_type (str): Compute type ("int8", "float16", "float32")
            
        Returns:
            bool: True if successful, False otherwise
        """
        new_config = {
            "model_type": model_type,
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type
        }
        return self.update_config(new_config)
