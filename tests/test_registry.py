"""
Tests for registry.py: load_registry, model_for_role, and related helpers.
No Ollama needed. Run: python -m pytest tests/
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from orchestrator.registry import (
    load_registry,
    model_for_role,
    default_text_model,
    candidates_for,
)

YAML_FIXTURE = textwrap.dedent("""\
    - name: coder
      backend: ollama
      tag: qwen2.5-coder:7b
      modality: text
      specialties: [code, debug]

    - name: general
      backend: ollama
      tag: gemma4:e2b
      modality: text
      specialties: [chat, summarize]
      default: true
      roles: [planner, synth]

    - name: image
      backend: diffusers
      tag: stabilityai/sd-turbo
      modality: image
      specialties: [image]
""")


@pytest.fixture()
def registry_file(tmp_path: Path) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(YAML_FIXTURE)
    return p


@pytest.fixture()
def reg(registry_file: Path):
    return load_registry(registry_file)


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------


def test_load_registry_parses_all_entries(reg):
    assert len(reg) == 3
    names = {m.name for m in reg}
    assert names == {"coder", "general", "image"}


def test_load_registry_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        load_registry(tmp_path / "nope.yaml")


def test_load_registry_empty_file_raises(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text("")
    with pytest.raises(RuntimeError, match="empty"):
        load_registry(p)


# ---------------------------------------------------------------------------
# candidates_for / default_text_model
# ---------------------------------------------------------------------------


def test_candidates_for_filters_modality(reg):
    text = candidates_for(reg, "text")
    assert all(m.modality == "text" for m in text)
    image = candidates_for(reg, "image")
    assert all(m.modality == "image" for m in image)


def test_default_text_model_returns_flagged_entry(reg):
    m = default_text_model(reg)
    assert m.name == "general"


# ---------------------------------------------------------------------------
# model_for_role
# ---------------------------------------------------------------------------


def test_model_for_role_planner(reg):
    m = model_for_role(reg, "planner")
    assert m.name == "general"


def test_model_for_role_synth(reg):
    m = model_for_role(reg, "synth")
    assert m.name == "general"


def test_model_for_role_unknown_falls_back_to_default(reg):
    m = model_for_role(reg, "nonexistent-role")
    assert m.default is True  # falls back to the default text model
