import requests
from typing import Optional, Dict, Any, BinaryIO

class ASRFusionClient:
    def __init__(self, base_url: str = "http://localhost:8603"):
        """
        Initialize ASR Fusion Client

        Args:
            base_url: Base URL of the ASR Fusion API server
        """
        self.base_url = base_url.rstrip('/')

    def transcribe_file(
        self,
        file_url: str,
        model: str = "faster-whisper/small",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        stream: bool = False,
        timestamp_granularities: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using the ASR Fusion API

        Args:
            file_path: Path to the audio file
            model: Model identifier in the format "engine/model_name"
            language: Language code (optional)
            prompt: Initial prompt for the transcription (optional)
            response_format: Response format (default: "json")
            temperature: Temperature for sampling (default: 0.0)
            timestamp_granularities: Timestamp granularities (optional)

        Returns:
            Transcription result
        """
        url = f"{self.base_url}/v1/audio/transcriptions"

        # Prepare files and data for the request
        data = {
            'file_url': file_url,
            'model': model,
            'response_format': response_format,
            'temperature': temperature
        }

        # Add optional parameters
        if language:
            data['language'] = language
        if prompt:
            data['prompt'] = prompt
        if timestamp_granularities:
            data['timestamp_granularities'] = timestamp_granularities

        # Make the request
        response = requests.post(url, data=data)
        response.raise_for_status()

        return response.json()
