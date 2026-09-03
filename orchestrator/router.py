"""
Router: pick the best model for one subtask.

Rule-based and deterministic: a `model_hint` set by the trained cost-aware router
(learned.py) wins if it names a model of the right modality; else match by
specialty tag; else the default text model. The trained part lives in learned.py
so this file stays a five-line lookup that tests can pin.
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
    if subtask.model_hint:  # trained cost-aware pick wins when it names a real model
        for m in pool:
            if m.name == subtask.model_hint:
                return m
    for m in pool:
        if subtask.type in m.specialties:
            return m
    if subtask.modality == "text":
        return default_text_model(registry)
    if pool:
        return pool[0]
    raise RuntimeError(f"No model configured for modality {subtask.modality!r}")
