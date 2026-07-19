"""
Synthesizer: merge all subtask results into one coherent final answer.

This is the Mixture-of-Agents "aggregator" idea: do not just concatenate the
pieces, ask a model to reconcile them into one clean answer. Skip the LLM call
entirely when there is only one result.
"""

from __future__ import annotations

from . import llm
from .schema import Result

SYNTH_PROMPT = """The user asked:
{query}

Here are answers to parts of that request:
{parts}

Combine them into ONE clear, correct, well-structured final answer. Resolve any
contradictions between the parts. Do not mention that the answer was assembled
from pieces.
"""


def _is_image(output: str) -> bool:
    # ponytail: Result has no modality field; image outputs are the only ones that
    # are a path under outputs/. Good enough to tell text from image here.
    return output.startswith("outputs/") or output.startswith("outputs\\")


def synthesize(query: str, results: list[Result], synth_model: str) -> str:
    """Merge per-subtask results into one answer; skip the LLM for a lone text result."""
    if len(results) == 1 and not _is_image(results[0].output):
        return results[0].output
    parts = []
    for r in results:
        body = f"[image saved at {r.output}]" if _is_image(r.output) else r.output
        parts.append(f"- (from {r.model_name}):\n{body}")
    prompt = SYNTH_PROMPT.format(query=query, parts="\n\n".join(parts))
    return llm.generate(synth_model, prompt)
