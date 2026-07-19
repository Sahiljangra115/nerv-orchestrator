"""
Tests for planner.py.  No Ollama needed: we patch llm.generate.
Run: python -m pytest tests/
"""

from __future__ import annotations

import json


from orchestrator import llm
from orchestrator.planner import plan, _fallback
from orchestrator.schema import Plan


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _valid_json(subtasks: list[dict]) -> str:
    return json.dumps({"subtasks": subtasks})


SINGLE_SUBTASK = [
    {
        "id": 0,
        "type": "chat",
        "modality": "text",
        "instruction": "hello",
        "depends_on": [],
        "hard": False,
    }
]

TWO_SUBTASKS = [
    {
        "id": 0,
        "type": "code",
        "modality": "text",
        "instruction": "write fib()",
        "depends_on": [],
        "hard": False,
    },
    {
        "id": 1,
        "type": "explain",
        "modality": "text",
        "instruction": "explain big-O",
        "depends_on": [0],
        "hard": False,
    },
]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_valid_json_returns_plan(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **kw: _valid_json(SINGLE_SUBTASK))
    p = plan("hello", "fake-model")
    assert isinstance(p, Plan)
    assert len(p.subtasks) == 1
    assert p.subtasks[0].type == "chat"


def test_multi_subtask_with_depends_on(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **kw: _valid_json(TWO_SUBTASKS))
    p = plan("write and explain fib", "fake-model")
    assert len(p.subtasks) == 2
    assert p.subtasks[1].depends_on == [0]


def test_invalid_json_triggers_retry_then_fallback(monkeypatch):
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append(1)
        return "not json at all"

    monkeypatch.setattr(llm, "generate", fake_generate)
    p = plan("anything", "fake-model", retries=2)
    # retries=2 means 3 total calls; each fails → fallback single-subtask plan
    assert len(calls) == 3
    assert len(p.subtasks) == 1
    assert p.subtasks[0].instruction == "anything"


def test_empty_subtasks_treated_as_failure(monkeypatch):
    """An empty subtasks list is technically valid JSON but useless; must fall back."""
    monkeypatch.setattr(llm, "generate", lambda *a, **kw: json.dumps({"subtasks": []}))
    p = plan("query", "fake-model", retries=0)
    assert len(p.subtasks) == 1


def test_fallback_returns_single_chat_subtask():
    p = _fallback("raw query")
    assert len(p.subtasks) == 1
    st = p.subtasks[0]
    assert st.type == "chat"
    assert st.instruction == "raw query"
    assert st.depends_on == []


if __name__ == "__main__":
    # quick self-check without pytest
    import types

    mock_llm = types.SimpleNamespace(
        generate=lambda *a, **kw: _valid_json(SINGLE_SUBTASK)
    )
    # can't monkeypatch without pytest here; just test _fallback
    p = _fallback("test query")
    assert p.subtasks[0].instruction == "test query"
    print("fallback test passed")
