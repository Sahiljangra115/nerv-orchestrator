"""
Tests for llm.py error paths.  No real Ollama needed: we patch urllib.request.urlopen.
Run: python -m pytest tests/
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest.mock import patch, MagicMock

import pytest

from orchestrator.llm import generate


def _fake_response(body: dict) -> MagicMock:
    """Returns a context-manager mock that reads JSON bytes."""
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_generate_returns_response_field():
    with patch(
        "urllib.request.urlopen", return_value=_fake_response({"response": "hello"})
    ):
        assert generate("model", "prompt") == "hello"


def test_generate_json_mode_adds_format_key():
    captured_data = []

    def fake_urlopen(req, timeout=None):
        captured_data.append(json.loads(req.data))
        return _fake_response({"response": "ok"})

    with patch("urllib.request.urlopen", fake_urlopen):
        generate("model", "prompt", json=True)

    assert captured_data[0].get("format") == "json"
    assert captured_data[0].get("stream") is False


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_http_error_mentions_model_and_pull(monkeypatch):
    def raise_http(*a, **kw):
        raise urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs={}, fp=None
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_http)
    with pytest.raises(RuntimeError, match="ollama pull"):
        generate("no-such-model:7b", "hi")


def test_url_error_mentions_ollama_serve(monkeypatch):
    def raise_url(*a, **kw):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    with pytest.raises(RuntimeError, match="ollama serve"):
        generate("model", "hi")


def test_timeout_error_is_friendly(monkeypatch):
    # socket read timeout raises TimeoutError, which is NOT a URLError, so it
    # must be caught separately or it escapes as a raw traceback.
    def raise_timeout(*a, **kw):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        generate("model", "hi")
