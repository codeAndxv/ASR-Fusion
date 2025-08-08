from typing import Dict, Any, Optional
from asr_fusion.config.config import Config
from asr_fusion.models.faster_whisper_model import FasterWhisperModel
from asr_fusion.models.funasr_model import FunASRModel
from asr_fusion.models.sensevoice_model import SenseVoiceModel

class ModelManager:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelManager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = Config(config_path)
        self.models = {}
    
    def load_model(self, model_identifier: str) -> Any:
        """
        Load a model based on the model identifier (e.g., "faster-whisper/large-v3")
        
        Args:
            model_identifier: Model identifier in the format "engine/model_name"
            
        Returns:
            Loaded model instance
        """
        if model_identifier in self.models:
            return self.models[model_identifier]
        
        # Parse the model identifier
        if "/" not in model_identifier:
            raise ValueError("Model identifier must be in the format 'engine/model_name'")
        
        engine, model_name = model_identifier.split("/", 1)
        
        # Get engine-specific configuration
        model_settings = self.config.get_model_config(engine, model_name)
        
        # Load the appropriate model
        if engine == "faster-whisper":
            model = FasterWhisperModel(
                model_name=model_name,
                model_path=model_settings.get("path", model_name),
                device=model_settings.get("device", "cpu"),
                compute_type=model_settings.get("compute_type", "int8")
            )
        elif engine == "funasr":
            model = FunASRModel(
                model_name=model_name,
                model_path=model_settings.get("path", model_name),
                device=model_settings.get("device", "cpu")
            )
        elif engine == "sensevoice":
            model = SenseVoiceModel(
                model_name=model_name,
                model_path=model_settings.get("path", model_name),
                device=model_settings.get("device", "cpu")
            )
        else:
            raise ValueError(f"Unsupported engine: {engine}")
        
        # Cache the model
        self.models[model_identifier] = model
        return model
    
    def transcribe_file(self, model_identifier: str, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file using the specified model
        
        Args:
            model_identifier: Model identifier in the format "engine/model_name"
            audio_file_path: Path to the audio file
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dictionary with transcription result
        """
        model = self.load_model(model_identifier)
        return model.transcribe_file(audio_file_path, **kwargs)
    
    def transcribe_stream(self, model_identifier: str, audio_chunks, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio stream using the specified model
        
        Args:
            model_identifier: Model identifier in the format "engine/model_name"
            audio_chunks: Audio chunks generator
            **kwargs: Additional arguments for transcription
            
        Returns:
            Dictionary with transcription result
        """
        model = self.load_model(model_identifier)
        return model.transcribe_stream(audio_chunks, **kwargs)

