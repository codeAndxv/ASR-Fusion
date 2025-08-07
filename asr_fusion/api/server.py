import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from asr_fusion.routers.transcription import router as transcription_router

app = FastAPI(title="ASR Fusion API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcription_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ASR Fusion API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    from asr_fusion.models.model_manager import ModelManager
    model_manager = ModelManager()
    config = model_manager.config.get_server_config()
    uvicorn.run(
        "asr_fusion.api.server:app",
        host=config.get("host", "localhost"),
        port=config.get("port", 8603),
        reload=True
    )
    print("server start in ")
