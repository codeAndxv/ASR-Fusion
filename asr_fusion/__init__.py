"""
ASR Fusion - A Python package for audio transcription using multiple ASR models
"""

__version__ = "0.1.0"
__author__ = "ASR Fusion Developers"

# Expose main classes and functions
from asr_fusion.models.model_manager import ModelManager
from asr_fusion.sdk.transcriber import ASRFusionClient

__all__ = [
    "ModelManager",
    "ASRFusionClient"
]
