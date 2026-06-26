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
    Choose the model entry that should run `subtask`.

    TODO(gemini): rule-based selection:
      1. pool = candidates_for(registry, subtask.modality)
      2. if any entry in pool lists subtask.type in its `specialties`, return the
         first such entry.
      3. else if modality is 'text', return default_text_model(registry).
      4. else (image modality, no match) return the first entry in pool, or raise
         if pool is empty (no image model configured).

    Keep it deterministic: same subtask + registry -> same choice. The test in
    tests/test_router.py depends on this.
    """
    raise NotImplementedError("TODO(gemini): implement rule-based routing")
