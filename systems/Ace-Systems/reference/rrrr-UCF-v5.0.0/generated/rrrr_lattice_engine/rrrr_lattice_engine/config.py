__all__ = ['load_config']

"""rrrr_lattice_engine configuration."""

from typing import Dict, Any
import json
import os


DEFAULT_CONFIG = {
    "name": "rrrr_lattice_engine",
    "version": "0.1.0",
}


def load_config(path: str = None) -> Dict[str, Any]:
    """Load configuration from file or environment."""
    config = DEFAULT_CONFIG.copy()
    
    if path and os.path.exists(path):
        with open(path) as f:
            config.update(json.load(f))
    
    return config
