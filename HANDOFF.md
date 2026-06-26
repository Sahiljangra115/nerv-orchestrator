# Handoff prompt for an implementing AI (e.g. Gemini 3.1 Pro)

Copy everything below the line into the other AI, with the repo attached.

---

You are implementing a Python project from a scaffold. Every file already exists
with a docstring contract and `# TODO(gemini): ...` markers. Your job: fill in the
TODOs so the project runs. Do NOT redesign the architecture or rename files.

## What the project does

A local-LLM orchestrator. It takes one user query, uses a "planner" model to split
it into subtasks, routes each subtask to the local model best suited for it, runs
them in dependency order (feeding earlier outputs into later ones), and synthesizes
one final answer. Text models run via Ollama's HTTP API; images via `diffusers`.

## Hard rules (do not violate)

1. `schema.py` is the source of truth and is already complete. Import its Pydantic
   models everywhere. Do not duplicate these shapes.
2. `llm.py` is the ONLY file allowed to know Ollama's API/URL. No other file may
   import `requests`/`httpx` or hardcode `localhost:11434`.
3. Every structured LLM output (the planner) must be Pydantic-validated, with retry
   on failure and a single-subtask fallback so the program never crashes on bad JSON.
4. Keep dependencies minimal: `pydantic`, `pyyaml`, `requests`, `pytest` for v1.
   Heavy imports (`torch`, `diffusers`, `treequest`) must live INSIDE the function
   that needs them, never at module top level.
5. Determinism where stated: `router.route` must return the same model for the same
   input. `tests/test_router.py` must pass unchanged.

## Implement in this order (each step is runnable)

- **Phase 0 - `llm.py`:** `generate(model, prompt, json=False)` POSTs to Ollama
  `/api/generate` (`stream=false`, add `format:"json"` when `json=True`) and returns
  the `response` text. Friendly error if Ollama is down.
- **Phase 1 - `registry.py` + `router.py`:** load `models.yaml` into `ModelEntry`
  objects; implement rule-based routing (modality -> specialty -> default). Make
  `tests/test_router.py` pass: `python -m pytest tests/`.
- **Phase 2 - `planner.py`:** `plan(query, planner_model)` returns a validated `Plan`.
  Use the validate+retry loop and `_fallback`. First make it return ONE subtask
  reliably, then enable multiple subtasks with `depends_on`.
- **Phase 3 - `executor.py`:** topological sort by `depends_on`, build prompts with
  dependency context, run each subtask, return `Result`s. (Leave `run_image` and
  `solve_hard` raising for now, or delegate `solve_hard` to a normal call.)
- **Phase 4 - `synthesizer.py`:** merge results into one answer; skip the call when
  there is a single text result.
- **Phase 5 - `cli.py`:** argparse, wire planner -> executor -> synthesizer, and
  print the trace (`[planner] N subtasks`, `[route] #id type -> model`, `[merge] ...`,
  then `--- final ---`). `python -m orchestrator "..."` must work end to end.
- **Phase 6 - `run_image` in `executor.py`:** generate via `diffusers`, save to
  `outputs/`, return the path. Note text/image GPU contention in a comment.
- **Phase 7 - `solve_hard` in `executor.py`:** AB-MCTS via `treequest` for subtasks
  flagged `hard` (generation fns per model + a 0..1 scoring fn). Optional / last.

## Acceptance

- `python -m pytest tests/` passes.
- `python -m orchestrator "write a python fibonacci function and explain its big-O"`
  prints a sensible trace (>=2 subtasks, code routed to the coder model) and a merged
  final answer.
- No file other than `llm.py` references the Ollama URL. No heavy imports at module
  top level.

Ask me before adding any dependency not in `requirements.txt`.
