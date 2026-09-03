"""
Learned planner + router: a small trained classifier replaces the LLM planner
for the common case, and picks the cheapest model that answers well enough.

Two heads, both sklearn pipelines over MiniLM sentence embeddings (22M params,
CPU, milliseconds per query), trained by `python -m trainer train` and pickled
to router.pkl at the repo root:

  type_clf  query -> subtask type (code/debug/explain/summarize/reason/chat)
  cost_clf  query -> registry model NAME. Label = cheapest model whose judged
            quality is within MARGIN of the best (see `cheapest` + trainer
            collect). Optional: only present once cost data was collected.

Question separation is a clause splitter (regex) + per-clause type prediction;
adjacent clauses of the same type are merged back. Deterministic.

Nothing here talks to Ollama. If router.pkl is missing, cli.py falls back to
the LLM planner in planner.py.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from .schema import Plan, Subtask

MODEL_PATH = Path(__file__).resolve().parent.parent / "router.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CONF = 0.35  # below this the type head is guessing; send to the default model
MARGIN = 0.1  # cost head: "good enough" = within this of the best judged score

_embedder = None
_bundle: dict | None = None


def embed(texts):
    """Sentence embeddings, unit-normalised. Module-level so the pickled pipeline
    can reference it by name (sklearn FunctionTransformer)."""
    global _embedder
    if _embedder is None:
        # ponytail: heavy import lives here, not at module top (HANDOFF rule 4).
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder.encode(list(texts), normalize_embeddings=True)


def available() -> bool:
    return MODEL_PATH.exists()


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with MODEL_PATH.open("rb") as f:
            _bundle = pickle.load(f)
    return _bundle


# ---------------------------------------------------------------------------
# question separation
# ---------------------------------------------------------------------------

# ponytail: regex clause splitter, not a parser. Splits on sentence ends, on
# ", then" / ", also", and on "and <imperative>" ("write X and explain Y").
# Ceiling: "e.g. foo" splits, and a single-sentence two-part ask with no verb
# cue stays whole. Upgrade path: --llm-plan (planner.py) for real DAGs.
_VERBS = (
    r"explain|describe|summari[sz]e|tell|show|reason|walk|write|implement|"
    r"debug|fix|list|compare|give|suggest|translate"
)
_SPLIT = re.compile(
    r"(?<=[.?!;])\s+(?=\S)"
    r"|,\s+(?:and\s+)?(?:then|also)\s+"
    rf"|\s+and\s+(?:then\s+|also\s+)?(?=(?:{_VERBS})\b)",
    re.IGNORECASE,
)
_LEAD = re.compile(
    r"^(?:then|also|and|next|finally|after that|afterwards)[\s,]+", re.IGNORECASE
)


def split_query(query: str) -> list[str]:
    """Split a query into candidate clauses. Tiny fragments (<3 words) glue onto
    the previous clause so trailing "Thanks." never becomes a subtask."""
    parts: list[str] = []
    for raw in _SPLIT.split(query.strip()):
        chunk = _LEAD.sub("", raw.strip())
        if not chunk:
            continue
        if parts and len(chunk.split()) < 3:
            parts[-1] = f"{parts[-1]} {chunk}"
        else:
            parts.append(chunk)
    return parts or [query.strip()]


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def classify(texts: list[str]) -> list[tuple[str, float]]:
    """Predict (type, confidence) per text with the type head."""
    clf = _load()["type_clf"]
    probs = clf.predict_proba(texts)
    return [(clf.classes_[row.argmax()], float(row.max())) for row in probs]


def pick_model(texts: list[str]) -> list[str | None]:
    """Predict the cheapest-good-enough registry model name per text, or None
    when no cost head has been trained yet (then the rule router decides)."""
    clf = _load().get("cost_clf")
    if clf is None:
        return [None] * len(texts)
    return [str(x) for x in clf.predict(texts)]


def plan(query: str) -> Plan:
    """Query -> Plan without any LLM call: split, classify each clause, merge
    same-type neighbours, chain dependencies, attach cost hints."""
    chunks = split_query(query)
    typed = classify(chunks)
    merged: list[tuple[str, str]] = []  # (type, instruction)
    for chunk, (t, conf) in zip(chunks, typed):
        t = t if conf >= MIN_CONF else "chat"
        if merged and merged[-1][0] == t:
            merged[-1] = (t, f"{merged[-1][1]} {chunk}")
        else:
            merged.append((t, chunk))
    hints = pick_model([m[1] for m in merged])
    # ponytail: naive chain (each step sees the previous output). Real DAGs need
    # the LLM planner; a chain is what "write X and explain it" wants anyway.
    return Plan(
        subtasks=[
            Subtask(
                id=i,
                type=t,
                instruction=text,
                depends_on=[i - 1] if i else [],
                model_hint=hint,
            )
            for i, ((t, text), hint) in enumerate(zip(merged, hints))
        ]
    )


# ---------------------------------------------------------------------------
# the routing objective (pure, used by trainer collect and tested)
# ---------------------------------------------------------------------------


def cheapest(per_model: dict[str, dict], margin: float = MARGIN) -> str | None:
    """
    per_model: {name: {"score": 0..1, "cost_ns": int}} (entries with an "error"
    key are skipped). Returns the name of the cheapest model whose score is
    within `margin` of the best score. That is the "cost to answer" objective:
    never pay for the big model when the small one is about as good.
    """
    ok = {n: v for n, v in per_model.items() if "score" in v}
    if not ok:
        return None
    best = max(v["score"] for v in ok.values())
    good = [n for n, v in ok.items() if v["score"] >= best - margin]
    return min(good, key=lambda n: ok[n]["cost_ns"])
