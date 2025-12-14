from pathlib import Path
import os

def get_project_root() -> Path:
    """Returns the project root directory."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "config" / "thresholds.json").exists():
            return parent
    return Path.cwd()  # Fallback

def get_config_path() -> Path:
    return get_project_root() / "config"

def get_model_path(model_name: str) -> Path:
    return get_project_root() / "models" / "saved" / model_name
