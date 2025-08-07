from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import tempfile
import os
from typing import Optional, List
from asr_fusion.models.model_manager import ModelManager

router = APIRouter(prefix="/v1/audio", tags=["audio"])

# Initialize model manager
model_manager = ModelManager()

@router.post("/transcriptions")
async def transcribe_file(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),
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
    if file is None and file_path is None:
        raise HTTPException(status_code=400, detail="Either 'file' or 'localfile_path' must be provided")
    
    # Validate that only one of file or localfile_path is provided
    if file is not None and file_path is not None:
        raise HTTPException(status_code=400, detail="Only one of 'file' or 'localfile_path' should be provided")
    
    # Validate response_format for specific models
    # Note: In a real implementation, we would check the model type
    # For now, we'll just validate the format is supported
    supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
    if response_format not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")
    
    # Validate timestamp_granularities when response_format is not verbose_json
    if timestamp_granularities and response_format != "verbose_json":
        raise HTTPException(status_code=400, detail="timestamp_granularities can only be used with response_format 'verbose_json'")
    
    # Validate timestamp_granularities values
    if timestamp_granularities:
        for granularity in timestamp_granularities:
            if granularity not in ["word", "segment"]:
                raise HTTPException(status_code=400, detail=f"Invalid timestamp_granularity: {granularity}")
    
    try:
        # Determine the audio file path to use
        if file is not None:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                audio_file_path = temp_file.name
            # Schedule cleanup of temporary file
            cleanup_files = [audio_file_path]
        else:
            # Use the provided local file path
            if not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"Local file not found: {file_path}")
            audio_file_path = file_path
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
        result = model_manager.transcribe_file(model, audio_file_path, **kwargs)
        
        # Clean up temporary files if any
        # for file_path in cleanup_files:
        #     try:
        #         os.unlink(file_path)
        #     except:
        #         pass  # Ignore cleanup errors
        
        # Return result in the requested format
        if response_format == "json":
            return result
        elif response_format == "text":
            # Concatenate all segments for text format
            text = "".join(segment["text"] for segment in result["segments"])
            return text
        elif response_format == "srt":
            # Generate SRT format
            srt_content = ""
            for i, segment in enumerate(result["segments"], 1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                srt_content += f"{i}\n{start} --> {end}\n{segment['text']}\n\n"
            return srt_content
        elif response_format == "verbose_json":
            return result
        elif response_format == "vtt":
            # Generate WebVTT format
            vtt_content = "WEBVTT\n\n"
            for segment in result["segments"]:
                start = format_timestamp(segment["start"], vtt=True)
                end = format_timestamp(segment["end"], vtt=True)
                vtt_content += f"{start} --> {end}\n{segment['text']}\n\n"
            return vtt_content
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_timestamp(seconds: float, vtt: bool = False) -> str:
    """Format timestamp in SRT or VTT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
