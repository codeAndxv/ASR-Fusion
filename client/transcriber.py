import requests
import json
from typing import Optional, Dict, Any, Generator

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
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        """
        Transcribe an audio file using the ASR Fusion API

        Args:
            file_url: URL to the audio file
            model: Model identifier in the format "engine/model_name"
            language: Language code (optional)
            prompt: Initial prompt for the transcription (optional)
            response_format: Response format (default: "json")
            temperature: Temperature for sampling (default: 0.0)
            stream: If True, stream the response (default: False)
            timestamp_granularities: Timestamp granularities (optional)

        Returns:
            Transcription result or generator for streaming
        """
        url = f"{self.base_url}/v1/audio/transcriptions"

        # Prepare files and data for the request
        data = {
            'file_url': file_url,
            'model': model,
            'response_format': response_format,
            'temperature': temperature,
            'stream': stream
        }

        # Add optional parameters
        if language:
            data['language'] = language
        if prompt:
            data['prompt'] = prompt
        if timestamp_granularities:
            data['timestamp_granularities'] = timestamp_granularities

        if stream:
            # Handle streaming response
            return self._stream_response(url, data)
        else:
            # Make the request
            response = requests.post(url, data=data)
            response.raise_for_status()

            return response.json()

    def _stream_response(self, url: str, data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Handle streaming response from the server

        Args:
            url: API endpoint URL
            data: Request data

        Yields:
            Decoded JSON objects from the streaming response
        """
        with requests.post(url, data=data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # Check if it's a data line (starts with "data: ")
                    if decoded_line.startswith("data: "):
                        # Extract JSON data after "data: "
                        json_data = decoded_line[6:]  # Remove "data: " prefix
                        try:
                            yield json.loads(json_data)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, yield raw data
                            yield {"raw": json_data}
