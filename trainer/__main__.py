"""
Train the small router that replaces the LLM planner.

    python -m trainer train              # type head (+ cost head if cost data exists)
    python -m trainer collect --limit 20 # run every text model on the question
                                         # pool via Ollama, judge, label cheapest
    python -m trainer eval               # held-out accuracy on bench/tasks.yaml
                                         # + clause-split demo + latency

Data: trainer/data.yaml (labelled questions). Cost labels: results/cost.jsonl
(append-only, resumable; `collect` skips questions already recorded).
Artifact: router.pkl at the repo root (gitignored; retrain takes seconds).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import yaml

from orchestrator import learned

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data.yaml"
COST = ROOT / "results" / "cost.jsonl"
BENCH_TASKS = ROOT / "bench" / "tasks.yaml"

# ponytail: cheap augmentation so the head is not thrown by "Please ..." / "Hey, ...".
PREFIXES = [
    "",
    "Can you ",
    "Please ",
    "Hey, ",
    "Quick one: ",
    "I need you to ",
    "Could you ",
]


def _pipeline(features: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = LogisticRegression(C=10, max_iter=2000)
    if features == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), clf
        )
    from sklearn.preprocessing import FunctionTransformer

    return make_pipeline(FunctionTransformer(learned.embed), clf)


def _augment(rows: list[dict], seed: int = 0) -> tuple[list[str], list[str], list[int]]:
    """Returns X, y, groups. groups = index of the source question, so CV never
    sees a prefixed copy of a question it is being tested on."""
    rng = random.Random(seed)
    X, y, g = [], [], []
    for i, r in enumerate(rows):
        q = r["q"]
        X.append(q)
        y.append(r["type"])
        g.append(i)
        for p in rng.sample(PREFIXES[1:], 2):
            X.append(p + q[0].lower() + q[1:])
            y.append(r["type"])
            g.append(i)
    return X, y, g


def _fit_report(
    name: str, pipe, X: list[str], y: list[str], groups: list[int] | None = None
) -> None:
    from sklearn.metrics import classification_report
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        StratifiedKFold,
        cross_val_predict,
    )

    counts = Counter(y)
    folds = min(5, min(counts.values()))
    if folds >= 2:
        if groups is None:
            cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        else:
            cv = StratifiedGroupKFold(folds, shuffle=True, random_state=0)
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
        acc = sum(p == t for p, t in zip(pred, y)) / len(y)
        print(f"[{name}] {len(y)} samples, {folds}-fold CV accuracy {acc:.3f}")
        print(classification_report(y, pred, zero_division=0))
    else:
        print(f"[{name}] {len(y)} samples, too few per class for CV: {dict(counts)}")
    pipe.fit(X, y)


def cmd_train(args: argparse.Namespace) -> None:
    rows = yaml.safe_load(DATA.read_text())
    X, y, groups = _augment(rows)
    type_clf = _pipeline(args.features)
    _fit_report("type", type_clf, X, y, groups)

    cost_clf = None
    cost_rows = _read_cost()
    labelled = [r for r in cost_rows if r.get("label")]
    if len(labelled) >= args.min_cost_rows:
        cost_clf = _pipeline(args.features)
        _fit_report(
            "cost", cost_clf, [r["q"] for r in labelled], [r["label"] for r in labelled]
        )
        print(f"[cost] label balance: {dict(Counter(r['label'] for r in labelled))}")
    else:
        print(
            f"[cost] {len(labelled)} labelled rows in {COST.name} "
            f"(< {args.min_cost_rows}); no cost head. Run: python -m trainer collect"
        )

    bundle = {
        "type_clf": type_clf,
        "cost_clf": cost_clf,
        "features": args.features,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_type": len(y),
        "n_cost": len(labelled),
    }
    with learned.MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"saved {learned.MODEL_PATH}")


def _read_cost() -> list[dict]:
    if not COST.exists():
        return []
    return [json.loads(line) for line in COST.read_text().splitlines() if line.strip()]


def cmd_collect(args: argparse.Namespace) -> None:
    """Run every text model on each pool question, judge each answer, label the
    cheapest good-enough model. Slow (one Ollama call per model per question,
    plus one judge call each); resumable, so run it in chunks with --limit."""
    from orchestrator import llm
    from orchestrator.executor import _score  # same judge the executor uses
    from orchestrator.registry import candidates_for, load_registry, model_for_role
    from orchestrator.schema import Subtask

    registry = load_registry()
    pool = candidates_for(registry, "text")
    judge = model_for_role(registry, "synth").tag
    rows = yaml.safe_load(DATA.read_text())
    done = {r["q"] for r in _read_cost()}
    todo = [r for r in rows if r["q"] not in done]
    if args.shuffle:
        random.Random(0).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    COST.parent.mkdir(exist_ok=True)
    print(
        f"{len(done)} done, {len(todo)} to collect x {len(pool)} models "
        f"(judge {judge}) -> {COST}"
    )
    for i, r in enumerate(todo, 1):
        q = r["q"]
        print(f"  [{i}/{len(todo)}] {r['type']}: {q[:60]}...", flush=True)
        per: dict[str, dict] = {}
        for m in pool:
            try:
                start = time.perf_counter()
                meta = llm.generate_meta(m.tag, q)
                secs = time.perf_counter() - start
                st = Subtask(id=0, type=r["type"], instruction=q)
                score = _score(st, meta["response"], judge)
            except RuntimeError as e:
                print(f"      {m.name}: ERROR {e}")
                per[m.name] = {"error": str(e)}
                continue
            # cost excludes model load time: cold starts are a hardware artefact,
            # not a property of the question.
            cost_ns = meta.get("total_duration", 0) - meta.get("load_duration", 0)
            per[m.name] = {
                "score": score,
                "cost_ns": int(cost_ns),
                "eval_count": meta.get("eval_count", 0),
                "seconds": round(secs, 1),
                "answer": meta["response"],
            }
            print(f"      {m.name}: score {score:.2f}  {cost_ns / 1e9:.1f}s")
        label = learned.cheapest(per, args.margin)
        print(f"      -> {label}")
        with COST.open("a") as f:
            f.write(
                json.dumps({"q": q, "type": r["type"], "models": per, "label": label})
                + "\n"
            )


def cmd_eval(args: argparse.Namespace) -> None:
    if not learned.available():
        raise SystemExit("no router.pkl. Run: python -m trainer train")
    tasks = yaml.safe_load(BENCH_TASKS.read_text())
    seeds = {r["q"] for r in yaml.safe_load(DATA.read_text())}
    tasks = [t for t in tasks if t["prompt"] not in seeds]  # keep it held out
    start = time.perf_counter()
    preds = learned.classify([t["prompt"] for t in tasks])
    warm = time.perf_counter() - start
    hits = 0
    for t, (p, conf) in zip(tasks, preds):
        ok = p == t["type"]
        hits += ok
        mark = "ok " if ok else "MISS"
        print(f"  {mark} {t['type']:<9} -> {p:<9} {conf:.2f}  {t['prompt'][:55]}")
    print(
        f"\nheld-out (bench/tasks.yaml): {hits}/{len(tasks)} = {hits / len(tasks):.2f}"
    )
    print(f"first call incl. model load: {warm:.1f}s")
    start = time.perf_counter()
    learned.plan("Write a Python fibonacci function and explain its big-O.")
    print(f"warm plan() latency: {(time.perf_counter() - start) * 1000:.0f}ms")

    print("\nsplit demo:")
    for q in [
        "Write a Python fibonacci function and explain its big-O.",
        "Debug this: def fib(n): return fib(n-1)+fib(n-2). Then explain why it fails.",
        "Summarize the CAP theorem, then write a toy key-value store in Python.",
        "What is the capital of France?",
    ]:
        p = learned.plan(q)
        print(f"  {q}")
        for st in p.subtasks:
            hint = f" [{st.model_hint}]" if st.model_hint else ""
            print(f"     #{st.id} {st.type}{hint} <- {st.depends_on}: {st.instruction}")

    b = learned._load()
    print(
        f"\nbundle: features={b['features']} n_type={b['n_type']} n_cost={b['n_cost']} "
        f"cost_head={'yes' if b.get('cost_clf') is not None else 'no'} trained {b['trained_at']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="trainer", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train type head (+ cost head if data exists)")
    t.add_argument("--features", choices=["minilm", "tfidf"], default="minilm")
    t.add_argument("--min-cost-rows", type=int, default=20)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("collect", help="collect cost/quality labels via Ollama")
    c.add_argument("--limit", type=int, default=0, help="questions this run (0 = all)")
    c.add_argument("--margin", type=float, default=learned.MARGIN)
    c.add_argument("--shuffle", action="store_true", help="mix types in a partial run")
    c.set_defaults(func=cmd_collect)

    e = sub.add_parser("eval", help="held-out accuracy + split demo")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
