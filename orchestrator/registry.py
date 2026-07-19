"""
The model registry: load models.yaml and answer questions about who is good at what.

Pure data + lookups. No LLM calls here. The router uses these helpers; it does not
parse YAML itself.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import ModelEntry

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models.yaml"


def load_registry(path: Path | None = None) -> list[ModelEntry]:
    """Read models.yaml into a list of validated ModelEntry objects."""
    path = path or DEFAULT_PATH
    if not path.exists():
        raise RuntimeError(f"Model registry not found at {path}")
    rows = yaml.safe_load(path.read_text())
    if not rows:
        raise RuntimeError(f"Model registry {path} is empty")
    return [ModelEntry(**row) for row in rows]


def candidates_for(registry: list[ModelEntry], modality: str) -> list[ModelEntry]:
    """Return entries matching the given modality ('text' or 'image')."""
    return [m for m in registry if m.modality == modality]


def default_text_model(registry: list[ModelEntry]) -> ModelEntry:
    """Return the fallback text model: the one flagged default, else the first text one."""
    text = candidates_for(registry, "text")
    for m in text:
        if m.default:
            return m
    if text:
        return text[0]
    raise RuntimeError("No text model in registry to fall back to")


def model_for_role(registry: list[ModelEntry], role: str) -> ModelEntry:
    """Return the first model tagged with `role` in its roles list; else the default text model."""
    for m in registry:
        if role in m.roles:
            return m
    return default_text_model(registry)
