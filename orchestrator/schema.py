"""
Data contracts for the whole system.

These Pydantic models are the SOURCE OF TRUTH. The planner produces a `Plan`,
the executor produces `Result`s, the router consumes `Subtask` + registry entries.
Every other module imports from here. This file is intentionally written out in
full (not stubbed) because it is the shared language between all the parts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Allowed subtask types. The planner must pick from this list, and the registry
# maps these to models. Keep this list and models.yaml `specialties` in sync.
SubtaskType = (
    str  # one of: "code", "debug", "explain", "summarize", "reason", "chat", "image"
)

Modality = str  # "text" | "image"


class Subtask(BaseModel):
    """One unit of work the planner carved out of the user's query."""

    id: int
    type: SubtaskType = "chat"
    modality: Modality = "text"
    instruction: str
    depends_on: list[int] = Field(
        default_factory=list
    )  # ids whose output feeds this one
    hard: bool = (
        False  # if True, executor may use the v2 search loop instead of a single call
    )
    # Registry model *name* chosen by the trained cost-aware router (learned.py).
    # None means "route by type". The rule router honours a hint that exists.
    model_hint: str | None = None


class Plan(BaseModel):
    """The planner's full decomposition of one query."""

    subtasks: list[Subtask]


class ModelEntry(BaseModel):
    """One row of the registry (one local model and what it is good at)."""

    name: str
    backend: str  # "ollama" | "diffusers"
    tag: str  # ollama model tag or diffusers model id
    modality: Modality = "text"
    specialties: list[str] = Field(default_factory=list)
    default: bool = False  # the fallback model when nothing else matches (text only)
    roles: list[str] = Field(default_factory=list)  # e.g. ["planner", "synth"]


class Result(BaseModel):
    """Output of running one subtask, plus the routing decision (for the trace)."""

    subtask_id: int
    model_name: str
    output: str  # text answer, or path to a saved image file
    seconds: float = 0.0
