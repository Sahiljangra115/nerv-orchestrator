"""
CLI entrypoint: wire planner -> executor -> synthesizer and PRINT THE TRACE.

The trace is the whole point of v1: you should be able to watch the system think
(what it split the query into, which model each part went to, how long each took)
before it prints the final merged answer.

Run with:  python -m orchestrator "your query here"
"""

from __future__ import annotations

import argparse

from .registry import load_registry
from .planner import plan
from .executor import execute
from .synthesizer import synthesize

# Which model drives planning and synthesis. Point these at a capable general
# model you have pulled. TODO(gemini): consider reading these from models.yaml
# (e.g. an entry flagged role: planner) instead of hardcoding.
PLANNER_MODEL = "gemma4:e2b"
SYNTH_MODEL = "gemma4:e2b"


def main(argv: list[str] | None = None) -> int:
    """
    TODO(gemini):
      - argparse: one positional `query`, optional --models PATH, optional --quiet.
      - registry = load_registry(models_path)
      - p = plan(query, PLANNER_MODEL); print "[planner] N subtasks".
      - for each subtask, the executor will route+run; print a "[route] #id type -> model"
        line per subtask (you can return routing info from execute or log inside it).
      - results = execute(p, registry)
      - print "[merge] synthesizing ...".
      - final = synthesize(query, results, SYNTH_MODEL)
      - print the final answer under a "--- final ---" header.
      - return 0 on success, non-zero on error (and print a friendly message if
        Ollama is not running).
    """
    raise NotImplementedError("TODO(gemini): wire the pipeline and print the trace")


if __name__ == "__main__":
    raise SystemExit(main())
