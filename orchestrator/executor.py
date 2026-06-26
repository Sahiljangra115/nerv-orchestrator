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

from . import llm
from .router import route
from .schema import ModelEntry, Plan, Result, Subtask


def execute(plan: Plan, registry: list[ModelEntry]) -> list[Result]:
    """
    Run the whole plan and return one Result per subtask.

    TODO(gemini):
      - order = topological_order(plan.subtasks)
      - results: dict[int, Result] = {}
      - for st in order:
          model = route(st, registry)
          context = "\\n\\n".join(results[d].output for d in st.depends_on)
          prompt = build_prompt(st, context)
          if st.hard:  output = solve_hard(st, registry)        # v2, stub for now
          elif st.modality == "image": output = run_image(model, prompt)
          else: output = llm.generate(model.tag, prompt)
          time the call and store Result(subtask_id=st.id, model_name=model.name, output=output, seconds=...)
      - return results in subtask-id order.
    """
    raise NotImplementedError("TODO(gemini): run subtasks in dependency order")


def topological_order(subtasks: list[Subtask]) -> list[Subtask]:
    """
    Return subtasks so every task comes after the ones it depends_on.

    TODO(gemini): simple Kahn's algorithm or DFS. For v1 you may assume ids are
    already roughly ordered, but do a real topo sort so depends_on is honored.
    Raise on a dependency cycle.
    """
    raise NotImplementedError("TODO(gemini): topological sort by depends_on")


def build_prompt(subtask: Subtask, context: str) -> str:
    """
    Combine a subtask's instruction with the outputs it depends on.

    TODO(gemini): if context is empty, return subtask.instruction; else prepend
    the dependency outputs under a clear header, e.g.
      "Context from previous steps:\\n{context}\\n\\nTask:\\n{instruction}".
    """
    raise NotImplementedError("TODO(gemini): assemble the prompt with context")


def run_image(model: ModelEntry, prompt: str) -> str:
    """
    Generate an image locally and return the saved file path. (Phase 6.)

    TODO(gemini): use diffusers (e.g. AutoPipelineForText2Image with model.tag,
    such as 'stabilityai/sd-turbo'), generate from `prompt`, save the PNG to a
    local 'outputs/' folder, and return the path string. Note: image and text
    models share GPU memory, so document/flag that the text model may need to be
    unloaded first. Keep heavy imports (torch, diffusers) INSIDE this function so
    text-only runs don't pay the import cost.
    """
    raise NotImplementedError("TODO(gemini): generate image with diffusers (Phase 6)")


def solve_hard(subtask: Subtask, registry: list[ModelEntry]) -> str:
    """
    v2 Sakana-style search for a single hard subtask. (Phase 7. Stub for now.)

    TODO(gemini, later): use the `treequest` library (AB-MCTS):
      - build generate_fns = {name: fn} backed by candidate models.
      - define a scoring function (0..1) for candidate answers:
          code  -> run it / pass a check; prose -> LLM-as-judge.
      - run algo.init_tree() then algo.step(...) for a fixed budget, return best.
    For v1, this can just delegate to a normal llm.generate so the field is wired
    but inert.
    """
    raise NotImplementedError("TODO(gemini): v2 AB-MCTS search (Phase 7)")
