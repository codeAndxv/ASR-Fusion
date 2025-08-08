from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import tempfile
import os
import json
from typing import Optional, List, Generator
from asr_fusion.models.model_manager import ModelManager

router = APIRouter(prefix="/v1/audio", tags=["audio"])

# Initialize model manager
model_manager = ModelManager()

@router.post("/transcriptions")
async def transcribe_file(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    model: str = Form("faster-whisper/large-v3"),
    chunking_strategy: Optional[str] = Form("auto"),
    include: Optional[List[str]] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    stream: Optional[bool] = Form(False),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(None)
):
    """
    Transcribes audio into the input language.
    
    Args:
        file: The audio file object (not file name) to transcribe
        localfile_path: Local file path to transcribe (alternative to file upload)
        model: ID of the model to use
        chunking_strategy: Controls how the audio is cut into chunks
        include: Additional information to include in the transcription response
        language: The language of the input audio
        prompt: An optional text to guide the model's style
        response_format: The format of the output
        stream: If set to true, the model response data will be streamed
        temperature: The sampling temperature
        timestamp_granularities: The timestamp granularities to populate
        
    Returns:
        Transcription result in the specified format
    """
    # Validate that either file or localfile_path is provided
    if file is None and file_url is None:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided")

    try:
        # Determine the audio file path to use
        if file is not None:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                audio_file_url = temp_file.name
            # Schedule cleanup of temporary file
            cleanup_files = [audio_file_url]
        else:
            # Use the provided local file path
            if file_url is not None and not os.path.exists(file_url):
                raise HTTPException(status_code=400, detail=f"Local file not found: {file_url}")
            audio_file_url = file_url
            cleanup_files = []
        
        # Prepare transcription arguments
        kwargs = {}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["initial_prompt"] = prompt
        kwargs["temperature"] = temperature
        if timestamp_granularities:
            kwargs["timestamp_granularities"] = timestamp_granularities
        
        # Perform transcription
        if stream:
            # Handle streaming response
            return StreamingResponse(
                stream_transcription(model, audio_file_url, **kwargs),
                media_type="text/event-stream"
            )
        else:
            result = model_manager.transcribe_file(model, audio_file_url, **kwargs)
            
            # Clean up temporary files if any
            # for file_path in cleanup_files:
            #     try:
            #         os.unlink(file_path)
            #     except:
            #         pass  # Ignore cleanup errors
            
            # Return result in the requested format
            if response_format == "json":
                return result
            elif response_format == "verbose_json":
                return result

    except HTTPException:
        raise
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))


def transcribe_file_to_streaming(model: str, audio_file_url: str, **kwargs) -> Generator[str, None, None]:
    """
    Stream transcription results in OpenAI format
    
    Args:
        model: Model identifier
        audio_file_url: Path to the audio file
        **kwargs: Additional arguments for transcription
        
    Yields:
        Formatted JSON strings in OpenAI streaming format
    """
    # Get streaming results from model manager
    stream_results = model_manager.transcribe_file_to_streaming(model, audio_file_url, **kwargs)
    
    for result in stream_results:
        # Format as data: JSON\n\n
        yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
