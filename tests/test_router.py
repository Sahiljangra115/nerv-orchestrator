"""
Deterministic spec for the router. The implementation in router.py must make
these pass. Run:  python -m pytest tests/  (or: python tests/test_router.py)

This is the contract: known type -> its specialist; unknown type -> default.
"""

from __future__ import annotations

from orchestrator.schema import ModelEntry, Subtask
from orchestrator.router import route

REGISTRY = [
    ModelEntry(
        name="coder",
        backend="ollama",
        tag="qwen2.5-coder:7b",
        modality="text",
        specialties=["code", "debug"],
    ),
    ModelEntry(
        name="general",
        backend="ollama",
        tag="gemma4:e2b",
        modality="text",
        specialties=["chat", "summarize", "explain", "reason"],
        default=True,
    ),
    ModelEntry(
        name="image",
        backend="diffusers",
        tag="stabilityai/sd-turbo",
        modality="image",
        specialties=["image"],
    ),
]


def test_code_subtask_goes_to_coder():
    st = Subtask(id=0, type="code", modality="text", instruction="write fib()")
    assert route(st, REGISTRY).name == "coder"


def test_unknown_type_falls_back_to_default():
    st = Subtask(id=0, type="poetry", modality="text", instruction="write a haiku")
    assert route(st, REGISTRY).name == "general"


def test_image_subtask_goes_to_image_model():
    st = Subtask(id=0, type="image", modality="image", instruction="a red fox")
    assert route(st, REGISTRY).name == "image"


if __name__ == "__main__":
    test_code_subtask_goes_to_coder()
    test_unknown_type_falls_back_to_default()
    test_image_subtask_goes_to_image_model()
    print("all router tests passed")
