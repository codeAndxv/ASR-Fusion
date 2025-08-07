import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[Any, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.config_data.get('server', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config_data.get('server', {}).get('model', {})
    
    def get_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Get specific model settings by name"""
        models = self.get_model_config()
        if isinstance(models, list):
            for model_entry in models:
                if isinstance(model_entry, dict):
                    for engine, model_configs in model_entry.items():
                        if isinstance(model_configs, list):
                            for model_config in model_configs:
                                if isinstance(model_config, dict):
                                    for model_key, settings in model_config.items():
                                        if model_key == model_name:
                                            return {
                                                "engine": engine,
                                                "settings": settings
                                            }
        return {}
