"""
Tests for the trained planner/router. No router.pkl and no MiniLM needed: the
splitter and the cost objective are pure, and plan() is tested with a fake head.
Run: python -m pytest tests/
"""

from __future__ import annotations

import numpy as np

from orchestrator import learned
from orchestrator.learned import cheapest, split_query
from orchestrator.router import route
from orchestrator.schema import ModelEntry, Subtask

REGISTRY = [
    ModelEntry(
        name="coder", backend="ollama", tag="q", modality="text", specialties=["code"]
    ),
    ModelEntry(
        name="general",
        backend="ollama",
        tag="g",
        modality="text",
        specialties=["chat"],
        default=True,
    ),
]


# ---------------------------------------------------------------------------
# question separation
# ---------------------------------------------------------------------------


def test_split_on_and_imperative():
    parts = split_query("Write a Python fibonacci function and explain its big-O.")
    assert parts == ["Write a Python fibonacci function", "explain its big-O."]


def test_split_on_sentence_and_strip_then():
    parts = split_query(
        "Debug this: def fib(n): return fib(n-1)+fib(n-2). Then explain why it fails."
    )
    assert len(parts) == 2
    assert parts[1].startswith("explain")


def test_short_fragment_glued_to_previous():
    assert split_query("Explain closures. Thanks!") == ["Explain closures. Thanks!"]


def test_single_query_untouched():
    assert split_query("What is the capital of France?") == [
        "What is the capital of France?"
    ]


# ---------------------------------------------------------------------------
# cost objective
# ---------------------------------------------------------------------------


def test_cheapest_prefers_small_model_within_margin():
    per = {
        "general": {"score": 0.9, "cost_ns": 50_000},
        "coder": {"score": 0.85, "cost_ns": 20_000},
    }
    assert cheapest(per, margin=0.1) == "coder"


def test_cheapest_pays_for_quality_outside_margin():
    per = {
        "general": {"score": 0.9, "cost_ns": 50_000},
        "coder": {"score": 0.5, "cost_ns": 20_000},
    }
    assert cheapest(per, margin=0.1) == "general"


def test_cheapest_skips_errors_and_handles_empty():
    assert cheapest({"a": {"error": "boom"}, "b": {"score": 0.3, "cost_ns": 1}}) == "b"
    assert cheapest({"a": {"error": "boom"}}) is None


# ---------------------------------------------------------------------------
# router honours the trained hint
# ---------------------------------------------------------------------------


def test_route_uses_model_hint():
    st = Subtask(id=0, type="chat", instruction="hi", model_hint="coder")
    assert route(st, REGISTRY).name == "coder"


def test_route_ignores_unknown_hint():
    st = Subtask(id=0, type="chat", instruction="hi", model_hint="ghost")
    assert route(st, REGISTRY).name == "general"


# ---------------------------------------------------------------------------
# plan() with a fake type head (no pickle, no embeddings)
# ---------------------------------------------------------------------------


class _FakeHead:
    classes_ = np.array(["code", "explain"])

    def predict_proba(self, texts):
        return np.array(
            [[0.9, 0.1] if "write" in t.lower() else [0.1, 0.9] for t in texts]
        )


def test_plan_chains_typed_subtasks(monkeypatch):
    monkeypatch.setattr(
        learned, "_load", lambda: {"type_clf": _FakeHead(), "cost_clf": None}
    )
    p = learned.plan("Write a fib function and explain its big-O.")
    assert [s.type for s in p.subtasks] == ["code", "explain"]
    assert p.subtasks[1].depends_on == [0]
    assert p.subtasks[0].model_hint is None


def test_plan_merges_same_type_neighbours(monkeypatch):
    monkeypatch.setattr(
        learned, "_load", lambda: {"type_clf": _FakeHead(), "cost_clf": None}
    )
    p = learned.plan("Write a stack class. Write push and pop for it.")
    assert len(p.subtasks) == 1 and p.subtasks[0].type == "code"
