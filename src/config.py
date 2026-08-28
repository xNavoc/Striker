"""
Configuration Loader Module
Parses settings.yaml into a structured Python dictionary.
"""

from pathlib import Path
import yaml


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Parses and returns the YAML configuration file.
    """
    full_path = get_project_root() / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
    
    with open(full_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


# Singleton configuration instance for imports
CONFIG = load_config()
