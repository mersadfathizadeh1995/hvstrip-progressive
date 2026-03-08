"""
Configuration management utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Optional, Union


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_configs(base: Dict, update: Dict) -> Dict:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict, output_path: Union[str, Path], format: str = 'yaml'):
    """Save configuration to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if format.lower() in ['yaml', 'yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


__all__ = [
    "load_config",
    "merge_configs", 
    "save_config"
]
