import json
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "model_type": "faster-whisper",  # faster-whisper or funasr
            "model_size": "small",  # tiny, base, small, medium, large
            "device": "cpu",  # cpu or cuda
            "compute_type": "int8"  # int8, float16, float32
        }
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge with default config to ensure all keys exist
                    merged_config = self.default_config.copy()
                    merged_config.update(config)
                    return merged_config
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Validate new config
            for key in new_config:
                if key not in self.default_config:
                    raise ValueError(f"Invalid configuration key: {key}")
            
            # Update config
            self.config.update(new_config)
            self.save_config(self.config)
            return True
        except Exception as e:
            print(f"Error updating config: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()
