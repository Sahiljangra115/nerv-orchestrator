"""
Tests for synthesizer.py.  No Ollama needed: we patch llm.generate.
Run: python -m pytest tests/
"""

from __future__ import annotations

from orchestrator import llm
from orchestrator.schema import Result
from orchestrator.synthesizer import synthesize, _is_image


def _result(subtask_id: int, model_name: str, output: str) -> Result:
    return Result(subtask_id=subtask_id, model_name=model_name, output=output)


# ---------------------------------------------------------------------------
# _is_image
# ---------------------------------------------------------------------------


def test_is_image_detects_outputs_prefix():
    assert _is_image("outputs/img_00123.png")
    assert _is_image("outputs\\img_00123.png")
    assert not _is_image("just some text")
    assert not _is_image("out/image.png")  # wrong prefix


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


def test_single_text_result_passes_through_without_llm(monkeypatch):
    called = []
    monkeypatch.setattr(llm, "generate", lambda *a, **kw: called.append(1) or "oops")
    results = [_result(0, "general", "the answer")]
    out = synthesize("q", results, "fake-model")
    assert out == "the answer"
    assert not called  # LLM must NOT be called for a lone text result


def test_single_image_result_calls_llm(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **kw: "merged answer")
    results = [_result(0, "image", "outputs/img_00001.png")]
    out = synthesize("q", results, "fake-model")
    assert out == "merged answer"


def test_multi_result_merge_prompt_contains_all_outputs(monkeypatch):
    captured = []

    def fake_generate(model, prompt, **kw):
        captured.append(prompt)
        return "merged"

    monkeypatch.setattr(llm, "generate", fake_generate)
    results = [
        _result(0, "coder", "def fib(): ..."),
        _result(1, "general", "O(n) time, O(1) space"),
    ]
    synthesize("write and explain fib", results, "fake-model")
    assert len(captured) == 1
    assert "def fib()" in captured[0]
    assert "O(n)" in captured[0]
    assert "write and explain fib" in captured[0]


def test_multi_result_image_entry_formatted_as_path(monkeypatch):
    captured = []
    monkeypatch.setattr(llm, "generate", lambda m, p, **kw: captured.append(p) or "x")
    results = [
        _result(0, "general", "some text"),
        _result(1, "image", "outputs/img_00001.png"),
    ]
    synthesize("q", results, "fake-model")
    assert "[image saved at outputs/img_00001.png]" in captured[0]
