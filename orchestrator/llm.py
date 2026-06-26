"""
The ONLY file that knows how to talk to Ollama.

Why isolated: backends change (Ollama today, vLLM or llama.cpp tomorrow). Keep
that knowledge in one place so the rest of the system never imports `requests`
or hardcodes a URL. If you swap backends, you rewrite only this file.

Contract: a single `generate()` function. Synchronous is fine for v1.
"""

from __future__ import annotations

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate(model: str, prompt: str, json: bool = False, timeout: float = 120.0) -> str:
    """
    Run one prompt against one local model and return the text response.

    TODO(gemini): implement with the stdlib or `requests`/`httpx`:
      - POST to OLLAMA_URL with body:
          {"model": model, "prompt": prompt, "stream": false}
        and add "format": "json" to the body when `json=True`.
      - parse the JSON response and return response["response"] (a str).
      - on connection error, raise a clear RuntimeError telling the user
        to start Ollama (`ollama serve`) and pull the model.
      - respect `timeout`.

    Notes:
      - `model` is the Ollama tag, e.g. "qwen2.5-coder:7b", NOT the registry name.
      - when json=True the caller will Pydantic-validate the string, so just
        return it raw here; do not parse it into a dict.
    """
    raise NotImplementedError("TODO(gemini): call Ollama and return the text response")
