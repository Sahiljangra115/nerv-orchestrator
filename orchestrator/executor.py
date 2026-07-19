"""
Executor: run every subtask in dependency order and collect results.

Responsibilities:
  - order subtasks so dependencies run first (topological order).
  - for each subtask: route it, build its prompt (injecting dependency outputs),
    run it on the right backend, store the Result.
  - text subtasks go through llm.generate; image subtasks go through run_image.

The v2 Sakana search loop (solve_hard) plugs in here: when subtask.hard is True,
replace the single llm.generate call with the tree-search version. The rest of
the system never sees the difference.
"""

from __future__ import annotations

import ast
import re
import time
from functools import partial
from typing import Callable

from . import llm
from .registry import candidates_for, default_text_model
from .router import route
from .schema import ModelEntry, Plan, Result, Subtask

# Phase 7 search budget: how many nodes AB-MCTS generates per hard subtask. Each
# node is one LLM call (plus a judge call), so keep it small on local models.
SEARCH_BUDGET = 8


def execute(
    plan: Plan,
    registry: list[ModelEntry],
    on_route: Callable[[Subtask, ModelEntry], None] | None = None,
) -> list[Result]:
    """
    Run the whole plan in dependency order and return one Result per subtask
    (sorted by id). `on_route`, if given, is called the moment each subtask is
    routed (before the slow call) so the CLI can stream the trace.
    """
    results: dict[int, Result] = {}
    for st in topological_order(plan.subtasks):
        model = route(st, registry)
        if on_route:
            on_route(st, model)
        context = "\n\n".join(results[d].output for d in st.depends_on if d in results)
        prompt = build_prompt(st, context)
        start = time.perf_counter()
        if st.hard:
            output = solve_hard(st, registry, context)
        elif st.modality == "image":
            output = run_image(model, prompt)
        else:
            output = llm.generate(model.tag, prompt)
        results[st.id] = Result(
            subtask_id=st.id,
            model_name=model.name,
            output=output,
            seconds=time.perf_counter() - start,
        )
    return [results[i] for i in sorted(results)]


def topological_order(subtasks: list[Subtask]) -> list[Subtask]:
    """
    Return subtasks so every task comes after the ones it depends_on (Kahn's
    algorithm). Ties broken by id for determinism. Raise on a dependency cycle.
    """
    # ponytail: O(n^2) re-scan per pop; plans are a handful of subtasks, not millions.
    by_id = {st.id: st for st in subtasks}
    indeg = {st.id: sum(1 for d in st.depends_on if d in by_id) for st in subtasks}
    ready = sorted(i for i, n in indeg.items() if n == 0)
    order: list[Subtask] = []
    while ready:
        i = ready.pop(0)
        order.append(by_id[i])
        for st in subtasks:
            if i in st.depends_on and indeg[st.id] > 0:
                indeg[st.id] -= 1
                if indeg[st.id] == 0:
                    ready.append(st.id)
        ready.sort()
    if len(order) != len(subtasks):
        raise RuntimeError("Dependency cycle in plan (subtasks depend on each other)")
    return order


def build_prompt(subtask: Subtask, context: str) -> str:
    """Prepend dependency outputs (if any) to the subtask instruction."""
    if not context:
        return subtask.instruction
    return f"Context from previous steps:\n{context}\n\nTask:\n{subtask.instruction}"


def run_image(model: ModelEntry, prompt: str) -> str:
    """
    Generate an image locally with diffusers and return the saved PNG path. (Phase 6.)
    """
    # ponytail: torch/diffusers imported here, not at module top, so text-only runs
    # stay light. GPU contention: the image model and Ollama's text model share VRAM;
    # `ollama stop <model>` to free it first if you OOM.
    import os

    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model.tag, torch_dtype=dtype).to(
        device
    )
    # sd-turbo is a 1-step, guidance-free model; these defaults suit it.
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", f"img_{abs(hash(prompt)) % 100000:05d}.png")
    image.save(path)
    return path


JUDGE_PROMPT = """Rate, as a single number from 0.0 (useless) to 1.0 (perfect), how
well the answer solves the task. Reply with ONLY the number.

Task:
{instruction}

Answer:
{answer}
"""


def solve_hard(
    subtask: Subtask,
    registry: list[ModelEntry],
    context: str = "",
    budget: int = SEARCH_BUDGET,
) -> str:
    """
    Phase 7: Sakana AB-MCTS ensemble search for one hard subtask (treequest).

    Each registry text model is an "action": the tree explores generating fresh
    answers and refining promising ones, scoring every node 0..1, and returns the
    best leaf. Falls back to a single normal call if `treequest` isn't installed.
    """
    # ponytail: treequest is heavy + optional, so import it here, not at module top.
    try:
        import treequest as tq
    except ImportError:
        # ponytail: degrade instead of crash if the v2 dep isn't installed.
        return llm.generate(
            route(subtask, registry).tag, build_prompt(subtask, context)
        )

    pool = candidates_for(registry, "text") or [default_text_model(registry)]
    judge = default_text_model(registry)
    base_prompt = build_prompt(subtask, context)

    def _generate(parent: dict | None, model_tag: str) -> tuple[dict, float]:
        if parent is None:
            prompt = base_prompt
        else:
            prompt = (
                f"{base_prompt}\n\nA previous attempt scored {parent['score']:.2f}:\n"
                f"{parent['answer']}\n\nProduce a better, corrected answer."
            )
        answer = llm.generate(model_tag, prompt)
        score = _score(subtask, answer, judge.tag)
        return {"answer": answer, "score": score}, score

    generate_fns = {m.name: partial(_generate, model_tag=m.tag) for m in pool}
    algo = tq.ABMCTSA()  # adaptive-branching MCTS, node-aggregation variant
    tree = algo.init_tree()
    for _ in range(budget):
        tree = algo.step(tree, generate_fns)
    best_state, _ = tq.top_k(tree, algo, k=1)[0]
    return best_state["answer"]


def _score(subtask: Subtask, answer: str, judge_tag: str) -> float:
    """Score a candidate answer 0..1: syntactic gate for code, then LLM-as-judge."""
    if subtask.type in ("code", "debug"):
        code = _extract_code(answer)
        # ponytail: parse-only check — we NEVER exec generated code (untrusted).
        if code and not _is_valid_python(code):
            return 0.0
    raw = llm.generate(
        judge_tag, JUDGE_PROMPT.format(instruction=subtask.instruction, answer=answer)
    )
    return _parse_score(raw)


def _extract_code(answer: str) -> str:
    """Pull the first fenced code block if present, else return the whole answer."""
    m = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", answer, re.DOTALL)
    return m.group(1) if m else answer


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _parse_score(text: str) -> float:
    """Extract the first number from the judge's reply, clamped to [0, 1]."""
    m = re.search(r"\d*\.?\d+", text)
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group())))
    except ValueError:
        return 0.0
