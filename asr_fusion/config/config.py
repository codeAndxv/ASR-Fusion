from configparser import ConfigParser

import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "../../config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[Any, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.config_data.get('server', {})

    def get_faster_whisper_config(self) -> Dict[str, Any]:
        return self.config_data.get('engine', {}).get('faster-whisper', {})

global_config = Config()