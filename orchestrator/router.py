"""
Router: pick the best model for one subtask.

v1 is deliberately dumb: match by modality, then by specialty tag, else fall back
to the default text model. This is rule-based on purpose. The smart upgrades
(embedding similarity, then a trained RouteLLM-style classifier) plug in HERE
later without touching planner/executor.
"""

from __future__ import annotations

from .schema import ModelEntry, Subtask
from .registry import candidates_for, default_text_model


def route(subtask: Subtask, registry: list[ModelEntry]) -> ModelEntry:
    """
    Choose the model entry that should run `subtask`. Rule-based and deterministic:
    specialty match within the modality wins; else the default text model; else
    (image, no match) the first image model.
    """
    pool = candidates_for(registry, subtask.modality)
    for m in pool:
        if subtask.type in m.specialties:
            return m
    if subtask.modality == "text":
        return default_text_model(registry)
    if pool:
        return pool[0]
    raise RuntimeError(f"No model configured for modality {subtask.modality!r}")
