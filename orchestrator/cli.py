"""
CLI entrypoint: wire planner -> executor -> synthesizer and PRINT THE TRACE.

The trace is the whole point of v1: you should be able to watch the system think
(what it split the query into, which model each part went to, how long each took)
before it prints the final merged answer.

Run with:  python -m orchestrator "your query here"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import learned
from .registry import load_registry, model_for_role
from .planner import plan
from .executor import execute
from .synthesizer import synthesize


def main(argv: list[str] | None = None) -> int:
    """Wire planner -> executor -> synthesizer and stream the trace to stdout."""
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Local LLM orchestrator: split, route, run, merge.",
    )
    parser.add_argument("query", help="the request to decompose and answer")
    parser.add_argument("--models", type=Path, default=None, help="path to models.yaml")
    parser.add_argument(
        "--quiet", action="store_true", help="hide the trace, print only the answer"
    )
    parser.add_argument("--planner", default=None, help="override planner model tag")
    parser.add_argument("--synth", default=None, help="override synthesizer model tag")
    parser.add_argument(
        "--llm-plan",
        action="store_true",
        help="use the LLM planner even when a trained router.pkl exists",
    )
    args = parser.parse_args(argv)

    def trace(line: str) -> None:
        if not args.quiet:
            print(line)

    try:
        registry = load_registry(args.models)
        planner_tag = args.planner or model_for_role(registry, "planner").tag
        synth_tag = args.synth or model_for_role(registry, "synth").tag
        if args.llm_plan or not learned.available():
            p = plan(args.query, planner_tag)
            trace(f"[planner] llm {len(p.subtasks)} subtasks")
        else:
            start = time.perf_counter()
            p = learned.plan(args.query)
            ms = (time.perf_counter() - start) * 1000
            trace(f"[planner] learned {len(p.subtasks)} subtasks ({ms:.0f}ms)")

        def on_route(st, model):
            how = " (cost)" if st.model_hint else ""
            trace(f"[route] #{st.id} {st.type} -> {model.name}{how}")

        results = execute(p, registry, on_route=on_route)
        for r in results:
            trace(f"[done]  #{r.subtask_id} {r.model_name} ({r.seconds:.1f}s)")
        trace("[merge] synthesizing ...")
        final = synthesize(args.query, results, synth_tag)
    except RuntimeError as e:
        print(f"error: {e}")
        return 1

    print("\n--- final ---")
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
