from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import json
from typing import Optional, Dict, Any
from asr_fusion.models.model_manager import ModelManager
from asr_fusion.config.config import Config

app = FastAPI(title="ASR Fusion API", description="Audio transcription API compatible with OpenAI's API format")

# Initialize config and model manager
config = Config("config.json")
model_manager = ModelManager(config)

@app.get("/")
async def root():
    return {"message": "ASR Fusion API is running", "model": model_manager.get_current_model_type()}

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: Optional[str] = Form(None)
):
    """
    Transcribe audio file - compatible with OpenAI's API
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Prepare transcription parameters
            kwargs = {}
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["hotword"] = prompt
            kwargs["temperature"] = temperature
            
            # Transcribe the file
            result = model_manager.transcribe_file(tmp_file_path, **kwargs)
            
            # Format response according to response_format
            if response_format == "text":
                return result["text"]
            elif response_format == "verbose_json":
                return JSONResponse(content=result)
            else:  # default to json
                return {"text": result["text"]}
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form("small"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0)
):
    """
    Translate audio file to English - compatible with OpenAI's API
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Prepare translation parameters (task=translate for Whisper models)
            kwargs = {
                "task": "translate",
                "temperature": temperature
            }
            if prompt:
                kwargs["hotword"] = prompt
            
            # Transcribe/translate the file
            result = model_manager.transcribe_file(tmp_file_path, **kwargs)
            
            # Format response according to response_format
            if response_format == "text":
                return result["text"]
            elif response_format == "verbose_json":
                return JSONResponse(content=result)
            else:  # default to json
                return {"text": result["text"]}
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/v1/config")
async def get_config():
    """
    Get current configuration
    """
    return config.get_all()

@app.post("/v1/config")
async def update_config(new_config: Dict[str, Any]):
    """
    Update configuration
    """
    try:
        success = model_manager.update_config(new_config)
        if success:
            return {"message": "Configuration updated successfully", "config": config.get_all()}
        else:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
