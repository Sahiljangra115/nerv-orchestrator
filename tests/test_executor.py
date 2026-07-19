"""
Spec for the executor's pure logic: dependency ordering and prompt assembly.
No Ollama needed (these never call a model). Run: python -m pytest tests/
"""

from __future__ import annotations

import pytest

from orchestrator.executor import (
    _extract_code,
    _is_valid_python,
    _parse_score,
    build_prompt,
    topological_order,
)
from orchestrator.schema import Subtask


def test_topo_orders_dependencies_first():
    subs = [
        Subtask(id=2, instruction="last", depends_on=[0, 1]),
        Subtask(id=0, instruction="first"),
        Subtask(id=1, instruction="middle", depends_on=[0]),
    ]
    order = [st.id for st in topological_order(subs)]
    # every dependency must appear before the task that needs it
    assert order.index(0) < order.index(1) < order.index(2)


def test_topo_raises_on_cycle():
    subs = [
        Subtask(id=0, instruction="a", depends_on=[1]),
        Subtask(id=1, instruction="b", depends_on=[0]),
    ]
    with pytest.raises(RuntimeError):
        topological_order(subs)


def test_build_prompt_with_and_without_context():
    st = Subtask(id=0, instruction="do the thing")
    assert build_prompt(st, "") == "do the thing"
    assert "earlier output" in build_prompt(st, "earlier output")
    assert "do the thing" in build_prompt(st, "earlier output")


def test_parse_score_clamps_and_extracts():
    assert _parse_score("0.7") == 0.7
    assert _parse_score("score: 0.42 out of 1") == 0.42
    assert _parse_score("5") == 1.0  # clamp above 1
    assert _parse_score("nonsense") == 0.0  # no number -> 0


def test_code_validity_and_extraction():
    assert _is_valid_python("def f():\n    return 1")
    assert not _is_valid_python("def f(:\n  return")
    fenced = "Here:\n```python\nx = 1\n```\nDone."
    assert _extract_code(fenced).strip() == "x = 1"
    assert _extract_code("no fence here") == "no fence here"


if __name__ == "__main__":
    test_topo_orders_dependencies_first()
    test_topo_raises_on_cycle()
    test_build_prompt_with_and_without_context()
    test_parse_score_clamps_and_extracts()
    test_code_validity_and_extraction()
    print("all executor tests passed")
