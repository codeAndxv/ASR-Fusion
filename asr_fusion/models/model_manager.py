from typing import Dict, Any, Optional
from asr_fusion.models.faster_whisper_model import FasterWhisperModel
from asr_fusion.models.funasr_model import FunASRModel
from asr_fusion.config.config import Config

class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self.current_model = None
        self.current_model_type = None
        self.load_model()

    def load_model(self) -> None:
        """Load the model based on configuration"""
        model_type = self.config.get("model_type", "faster-whisper")
        
        # If the model type hasn't changed, no need to reload
        if self.current_model and self.current_model_type == model_type:
            return
        
        # Load the appropriate model
        if model_type == "faster-whisper":
            self.current_model = FasterWhisperModel(self.config.get_all())
            self.current_model_type = "faster-whisper"
            print("Loaded faster-whisper model")
        elif model_type == "funasr":
            self.current_model = FunASRModel(self.config.get_all())
            self.current_model_type = "funasr"
            print("Loaded FunASR model")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def transcribe_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file using the current model"""
        if not self.current_model:
            self.load_model()
        
        return self.current_model.transcribe_file(audio_file_path, **kwargs)

    def transcribe_stream(self, audio_stream, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio stream using the current model"""
        if not self.current_model:
            self.load_model()
        
        return self.current_model.transcribe_stream(audio_stream, **kwargs)

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration and reload model if necessary"""
        success = self.config.update_config(new_config)
        if success:
            # Reload model if model type or other relevant config changed
            self.load_model()
        return success

    def get_current_model_type(self) -> str:
        """Get the current model type"""
        return self.current_model_type if self.current_model_type else "none"
