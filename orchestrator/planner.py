"""
Planner: turn a free-text query into a validated Plan of subtasks.

This is the hardest module. The difficulty is NOT the prompt, it is making an LLM
return JSON that actually matches the schema, every time. The pattern that works:
ask with format=json -> validate with Pydantic -> on failure, retry feeding the
error back -> after N failures, fall back to a single subtask so we never crash.

Build it in two stages while implementing:
  Stage A: get it to reliably return ONE subtask.
  Stage B: then allow multiple subtasks with depends_on.
"""

from __future__ import annotations

from . import llm
from .schema import Plan, Subtask

ALLOWED_TYPES = ["code", "debug", "explain", "summarize", "reason", "chat", "image"]

PLANNER_PROMPT = """You are a task planner. Split the user's request into the
SMALLEST useful set of subtasks. Return ONLY JSON matching this shape:

{{"subtasks": [
  {{"id": 0, "type": "<one of {allowed}>", "modality": "text|image",
    "instruction": "<what to do>", "depends_on": [<ids>], "hard": false}}
]}}

Rules:
- ids start at 0 and increase.
- use depends_on when a subtask needs an earlier subtask's output.
- use modality "image" only for image-generation subtasks.
- mark hard=true only for genuinely difficult subtasks (math proofs, tricky code).
- if the request is simple, return a single subtask of type "chat".

User request:
{query}
"""


def plan(query: str, planner_model: str, retries: int = 2) -> Plan:
    """
    Decompose `query` into a validated Plan using the given Ollama model tag.

    TODO(gemini):
      - loop up to retries+1 times:
          raw = llm.generate(planner_model, PLANNER_PROMPT.format(allowed=ALLOWED_TYPES, query=query), json=True)
          try Plan.model_validate_json(raw) and return it on success.
          on pydantic ValidationError, append the error text to the prompt/query
          and try again ("your last output was invalid: ... fix it").
      - if all attempts fail, return the single-subtask fallback below so the
        system degrades instead of throwing.
    """
    raise NotImplementedError("TODO(gemini): implement validate+retry planning")


def _fallback(query: str) -> Plan:
    """Single-subtask plan used when the planner cannot produce valid JSON."""
    return Plan(subtasks=[Subtask(id=0, type="chat", modality="text", instruction=query)])
