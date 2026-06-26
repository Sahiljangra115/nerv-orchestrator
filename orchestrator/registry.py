"""
The model registry: load models.yaml and answer questions about who is good at what.

Pure data + lookups. No LLM calls here. The router uses these helpers; it does not
parse YAML itself.
"""

from __future__ import annotations

from pathlib import Path

from .schema import ModelEntry

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models.yaml"


def load_registry(path: Path | None = None) -> list[ModelEntry]:
    """
    Read models.yaml into a list of validated ModelEntry objects.

    TODO(gemini):
      - default `path` to DEFAULT_PATH.
      - parse the YAML (PyYAML `yaml.safe_load`) which is a list of dicts.
      - validate each dict with ModelEntry(**d).
      - raise a clear error if the file is missing or empty.
    """
    raise NotImplementedError("TODO(gemini): load and validate models.yaml")


def candidates_for(registry: list[ModelEntry], modality: str) -> list[ModelEntry]:
    """Return entries matching the given modality ('text' or 'image'). One-liner."""
    raise NotImplementedError("TODO(gemini): filter registry by modality")


def default_text_model(registry: list[ModelEntry]) -> ModelEntry:
    """
    Return the fallback text model.

    TODO(gemini): return the entry with default=True; if none is marked,
    return the first text entry; if there are no text entries, raise.
    """
    raise NotImplementedError("TODO(gemini): pick the default text model")
