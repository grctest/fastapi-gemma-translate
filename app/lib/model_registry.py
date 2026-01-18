from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _find_model_files(models_dir: Path) -> Dict[str, Path]:
    model_map: Dict[str, Path] = {}

    if not models_dir.exists():
        return model_map

    for folder in models_dir.iterdir():
        if not folder.is_dir():
            continue

        gguf_files = sorted(folder.glob("*.gguf"))
        if not gguf_files:
            continue

        model_map[folder.name] = gguf_files[0]

    return model_map


def get_available_models() -> Dict[str, Path]:
    return _find_model_files(MODELS_DIR)


def build_model_enum() -> Enum:
    models = get_available_models()
    if not models:
        return Enum("ModelName", {"NO_MODELS_FOUND": "NO_MODELS_FOUND"})

    return Enum("ModelName", {name: name for name in models.keys()})


def resolve_model_path(model_name: str) -> Path:
    models = get_available_models()
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in {MODELS_DIR}")

    return models[model_name]


def list_model_names() -> List[str]:
    return sorted(get_available_models().keys())


def list_models_with_paths() -> List[Tuple[str, str]]:
    return [(name, str(path)) for name, path in get_available_models().items()]