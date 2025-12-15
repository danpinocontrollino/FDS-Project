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

def add_project_root_to_sys_path():
    """Insert the project root at the front of sys.path for imports."""
    import sys
    root = get_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
