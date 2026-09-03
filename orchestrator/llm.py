"""
The ONLY file that knows how to talk to Ollama.

Why isolated: backends change (Ollama today, vLLM or llama.cpp tomorrow). Keep
that knowledge in one place so the rest of the system never imports `requests`
or hardcodes a URL. If you swap backends, you rewrite only this file.

Contract: a single `generate()` function. Synchronous is fine for v1.
"""

from __future__ import annotations

import json as _json  # aliased so the `json` parameter below can keep its contract name
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate(
    model: str, prompt: str, json: bool = False, timeout: float = 300.0
) -> str:
    """
    Run one prompt against one local model and return the text response.

    `model` is the Ollama tag (e.g. "qwen2.5-coder:7b"), not the registry name.
    When json=True we ask Ollama for JSON and return it raw; the caller validates.
    """
    return generate_meta(model, prompt, json=json, timeout=timeout)["response"]


def generate_meta(
    model: str, prompt: str, json: bool = False, timeout: float = 300.0
) -> dict:
    """
    Like generate() but return Ollama's whole payload: `response` plus timing and
    token counts (`total_duration`, `load_duration`, `eval_count`, ... in ns).
    The trainer's cost collection needs those numbers; nothing else should.
    """
    # ponytail: stdlib urllib does a plain POST fine, so no `requests` dependency.
    body = {"model": model, "prompt": prompt, "stream": False}
    if json:
        body["format"] = "json"
    req = urllib.request.Request(
        OLLAMA_URL,
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Ollama IS reachable but rejected the request. A 404 here means the model
        # tag isn't pulled. Catch this BEFORE URLError (HTTPError subclasses it).
        raise RuntimeError(
            f"Ollama returned HTTP {e.code} for model {model!r}. "
            f"If it is not pulled yet, run: ollama pull {model}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} ({e.reason}). "
            "Is it running? Start it with `ollama serve`."
        ) from e
    except TimeoutError as e:
        # Read timeout: Ollama accepted the request but didn't respond in time.
        # Common on the first call to a cold model (VRAM load). Not a URLError.
        raise RuntimeError(
            f"Ollama timed out after {timeout:.0f}s for model {model!r}. "
            "A cold model can be slow on first load; retry, or raise the timeout."
        ) from e
    return payload
