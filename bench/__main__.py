"""
Benchmark harness: NERV pipeline vs. single-model baseline.

Usage:
    python -m bench run          # run all tasks, save results/
    python -m bench report       # print win-rate table from last run

Each task in tasks.yaml is run twice:
  A  – NERV full pipeline (plan → route → execute → synthesize)
  B  – single default model, raw prompt (no orchestration)

A judge model scores each pair (0–1) with randomised A/B order to reduce
position bias. 3 repeats, majority vote determines the winner.

Results are written as JSON lines to results/<timestamp>.jsonl so every run
is preserved and replayable.

Notes:
  - Needs Ollama running: ollama serve
  - Judge is the default text model (same model used for B baseline).
    If that model wins A, the single-model baseline beats orchestration —
    that is a valid finding, not a failure. Publish the numbers honestly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import yaml

# ponytail: stdlib only, no new deps
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TASKS_FILE = Path(__file__).resolve().parent / "tasks.yaml"

JUDGE_PROMPT = """\
You are a fair judge. Two answers are given to the same question.
Rate which is better: reply with ONLY "A", "B", or "tie".

Question: {question}

Answer A:
{a}

Answer B:
{b}
"""


def _run_nerv(
    query: str, registry, planner_model_tag: str, synth_model_tag: str
) -> tuple[str, float]:
    from orchestrator.planner import plan
    from orchestrator.executor import execute
    from orchestrator.synthesizer import synthesize

    start = time.perf_counter()
    p = plan(query, planner_model_tag)
    results = execute(p, registry)
    answer = synthesize(query, results, synth_model_tag)
    return answer, time.perf_counter() - start


def _run_baseline(query: str, model_tag: str) -> tuple[str, float]:
    from orchestrator import llm

    start = time.perf_counter()
    answer = llm.generate(model_tag, query)
    return answer, time.perf_counter() - start


def _judge(question: str, a: str, b: str, judge_tag: str) -> str:
    """Returns 'A', 'B', or 'tie'."""
    from orchestrator import llm

    prompt = JUDGE_PROMPT.format(question=question, a=a, b=b)
    raw = llm.generate(judge_tag, prompt).strip().upper()
    if raw.startswith("A"):
        return "A"
    if raw.startswith("B"):
        return "B"
    return "tie"


def _majority(votes: list[str]) -> str:
    from collections import Counter

    (winner, _), *_ = Counter(votes).most_common(1)
    return winner


def cmd_run(args: argparse.Namespace) -> None:
    from orchestrator.registry import load_registry, model_for_role

    registry = load_registry()
    planner_tag = model_for_role(registry, "planner").tag
    synth_tag = model_for_role(registry, "synth").tag
    baseline_model = model_for_role(
        registry, "synth"
    )  # same capable model for baseline
    judge_tag = baseline_model.tag

    tasks = yaml.safe_load(TASKS_FILE.read_text())
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_file = RESULTS_DIR / f"{ts}.jsonl"

    repeats = args.repeats
    print(f"Running {len(tasks)} tasks × {repeats} repeats each → {out_file}")
    nerv_wins = baseline_wins = ties = 0

    for i, task in enumerate(tasks, 1):
        query = task["prompt"]
        task_type = task.get("type", "unknown")
        print(f"  [{i}/{len(tasks)}] {task_type}: {query[:60]}...", end=" ", flush=True)

        nerv_ans, nerv_s = _run_nerv(query, registry, planner_tag, synth_tag)
        base_ans, base_s = _run_baseline(query, judge_tag)

        votes = []
        for _ in range(repeats):
            # randomise A/B assignment to reduce position bias
            flip = random.random() < 0.5
            a, b = (nerv_ans, base_ans) if not flip else (base_ans, nerv_ans)
            raw = _judge(query, a, b, judge_tag)
            if raw == "tie":
                votes.append("tie")
            elif (raw == "A" and not flip) or (raw == "B" and flip):
                votes.append("nerv")
            else:
                votes.append("baseline")

        winner = _majority(votes)
        if winner == "nerv":
            nerv_wins += 1
        elif winner == "baseline":
            baseline_wins += 1
        else:
            ties += 1
        print(f"→ {winner}  (nerv {nerv_s:.1f}s  baseline {base_s:.1f}s)")

        record = {
            "task_type": task_type,
            "query": query,
            "nerv_answer": nerv_ans,
            "nerv_seconds": round(nerv_s, 2),
            "baseline_answer": base_ans,
            "baseline_seconds": round(base_s, 2),
            "votes": votes,
            "winner": winner,
        }
        with out_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

    total = nerv_wins + baseline_wins + ties
    print(
        f"\nResults: NERV {nerv_wins}/{total}  baseline {baseline_wins}/{total}  ties {ties}/{total}"
    )
    print(f"Saved to {out_file}")


def cmd_report(args: argparse.Namespace) -> None:
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print("No results found. Run: python -m bench run")
        return
    path = files[-1]
    print(f"Report from {path.name}\n")

    from collections import defaultdict

    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"nerv": 0, "baseline": 0, "tie": 0}
    )
    totals = {"nerv": 0, "baseline": 0, "tie": 0}

    for line in path.read_text().splitlines():
        r = json.loads(line)
        w = r["winner"]
        by_type[r["task_type"]][w] += 1
        totals[w] += 1

    print(f"{'type':<12} {'NERV':>6} {'base':>6} {'tie':>6} {'total':>7}")
    print("-" * 40)
    for ttype, counts in sorted(by_type.items()):
        n, b, t = counts["nerv"], counts["baseline"], counts["tie"]
        print(f"{ttype:<12} {n:>6} {b:>6} {t:>6} {n + b + t:>7}")
    print("-" * 40)
    n, b, t = totals["nerv"], totals["baseline"], totals["tie"]
    print(f"{'TOTAL':<12} {n:>6} {b:>6} {t:>6} {n + b + t:>7}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench", description="NERV benchmark harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run all benchmark tasks")
    run_p.add_argument(
        "--repeats", type=int, default=3, help="Judge repeats per task (default 3)"
    )
    run_p.set_defaults(func=cmd_run)

    report_p = sub.add_parser("report", help="Print win-rate table from last run")
    report_p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
