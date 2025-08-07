from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import os
from typing import Optional
from asr_fusion.models.model_manager import ModelManager

router = APIRouter(prefix="/v1/audio", tags=["audio"])

# Initialize model manager
model_manager = ModelManager()

@router.post("/transcriptions")
async def transcribe_file(
    file: UploadFile = File(...),
    model: str = Form("faster-whisper/large-v3"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None)
):
    """
    Transcribe an audio file
    
    Args:
        file: Audio file to transcribe
        model: Model identifier in the format "engine/model_name"
        language: Language code (optional)
        prompt: Initial prompt for the transcription (optional)
        response_format: Response format (default: "json")
        temperature: Temperature for sampling (default: 0.0)
        timestamp_granularities: Timestamp granularities (optional)
        
    Returns:
        Transcription result in the specified format
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
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
        result = model_manager.transcribe_file(model, temp_file_path, **kwargs)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
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
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")
            
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
