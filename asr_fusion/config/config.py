from configparser import ConfigParser

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

    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific engine"""
        return self.config_data.get('engine', {}).get(engine_name, {})

    def get_model_config(self, engine_name: str, model_name: str) -> Dict[str, Any]:
        return self.config_data.get('model', {}).get(engine_name, {}).get(model_name, {})

global_config = Config()
