# NERV

*Local LLM Agent Orchestrator. Named after NERV (Evangelion): one command center
that directs each specialized unit to the job it is best at.*

Take one query, split it into parts, send each part to whichever local model is
best at it, then stitch the answers back together. Runs on local models via
[Ollama](https://ollama.com) (text) and `diffusers` (images). No cloud APIs.

> ✅ **v1 complete.** Phases 0–5 implemented and tested. Phase 6 (image/diffusers)
> and Phase 7 (AB-MCTS search via `treequest`) are wired but gated behind optional
> deps — they degrade gracefully without them.

---

## 30-second quickstart

```bash
pip install -r requirements.txt

# In a separate terminal:
ollama serve

# Pull the two text models:
ollama pull gemma4:e2b
ollama pull qwen2.5-coder:7b

python -m pytest tests/
python -m orchestrator "write a Python fibonacci function and explain its big-O"
```

---

## Example trace

```
[planner] 2 subtasks
[route] #0 code -> coder
[route] #1 explain -> general
[done]  #0 coder (4.2s)
[done]  #1 general (3.8s)
[merge] synthesizing ...

--- final ---
Here is a Python fibonacci function and an explanation of its complexity:

def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

**Time complexity:** O(n) — the loop runs exactly n times.
**Space complexity:** O(1) — only two variables are kept in memory at once.
...
```

---

## Architecture

```
query -> PLANNER (LLM, format=json, Pydantic validate, retry x2,
         fallback single-chat plan, never crashes)
      -> subtasks with depends_on
      -> Kahn topological order:
           ROUTER  -> specialty match within modality
                      else default text model
                      else first image model
           EXECUTOR -> build prompt with dependency context
                       hard=True  -> solve_hard (AB-MCTS, inert without treequest)
                       image      -> run_image (diffusers, inert without torch)
                       else       -> llm.generate (Ollama, stdlib urllib)
      -> SYNTHESIZER -> lone text result passes through raw (no extra LLM call)
                        else LLM merge prompt
      -> final answer + [planner]/[route]/[done]/[merge] trace
```

Five modules around a shared Pydantic schema:

| Module            | Responsibility                                              |
|-------------------|-------------------------------------------------------------|
| `schema.py`       | Single source of truth: `Subtask`, `Plan`, `ModelEntry`, `Result` |
| `llm.py`          | The **only** file that knows Ollama's URL / HTTP API        |
| `registry.py`     | Load `models.yaml`; lookups by modality, role, specialty    |
| `planner.py`      | Query → validated `Plan` (validate + retry + fallback)      |
| `router.py`       | `Subtask` + registry → `ModelEntry` (deterministic, tested) |
| `executor.py`     | Topological order → run → collect `Result`s                 |
| `synthesizer.py`  | Merge results into one answer                               |
| `cli.py`          | Argparse entry point, streams the trace                     |

---

## Layout

```
models.yaml            registry: model -> {specialties, modality, roles, tag}
orchestrator/
  schema.py            Pydantic contracts (the shared language)
  llm.py               the only file that talks to Ollama
  registry.py          load models.yaml, lookups
  planner.py           query -> validated Plan (validate + retry)
  router.py            subtask -> model (rule-based, deterministic)
  executor.py          run subtasks in dependency order
  synthesizer.py       merge results into one answer
  cli.py               entrypoint, prints the trace
tests/
  test_router.py       deterministic routing spec
  test_executor.py     topology, prompt building, score parsing (no Ollama)
  test_planner.py      validate/retry/fallback paths (no Ollama)
  test_synthesizer.py  passthrough + merge logic (no Ollama)
  test_llm.py          HTTP/URL error messages (no Ollama)
  test_registry.py     load, roles, default model lookups
bench/
  tasks.yaml           20 benchmark prompts (code/debug/explain/summarize/reason)
  __main__.py          NERV vs single-model harness; run: python -m bench run
```

---

## CLI flags

```
python -m orchestrator "query"
  --models PATH     path to a custom models.yaml (default: repo root)
  --quiet           hide the trace, print only the final answer
  --planner TAG     override planner model tag (overrides models.yaml roles)
  --synth TAG       override synthesizer model tag
```

---

## models.yaml

Edit this to match the models you have pulled. A model tagged with a `roles` entry
is auto-selected by the CLI for that role — no hardcoding needed:

```yaml
- name: general
  backend: ollama
  tag: gemma4:e2b
  modality: text
  specialties: [chat, summarize, explain, reason]
  default: true
  roles: [planner, synth]   # drives planning and synthesis
```

---

## Benchmark harness

```bash
ollama serve   # in another terminal
python -m bench run        # runs 20 tasks × 3 judge repeats, saves results/
python -m bench report     # prints win-rate table by subtask type
```

NERV vs single-model baseline judged by the strongest local model you have, with
randomised A/B ordering to reduce position bias. Honest negative results (where the
single model wins) are just as valuable as positive ones.

---

## Optional extras

| Extra        | Install                                | What it enables                          |
|--------------|----------------------------------------|------------------------------------------|
| Image gen    | `pip install torch diffusers accelerate` | `run_image` in executor (Phase 6)        |
| AB-MCTS      | `pip install treequest` (Python 3.11+) | `solve_hard` in executor (Phase 7)       |

Both degrade gracefully when not installed: image subtasks and hard subtasks fall
back to a normal `llm.generate` call.

---

## Hard rules (from HANDOFF.md)

1. `schema.py` is the source of truth. Import its Pydantic models everywhere.
2. `llm.py` is the **only** file allowed to know Ollama's API/URL.
3. Every structured LLM output is Pydantic-validated with retry and a single-subtask
   fallback — the program never crashes on bad JSON.
4. Heavy imports (`torch`, `diffusers`, `treequest`) live **inside** the function that
   needs them, never at module top level.
5. `router.route` is deterministic: same input, same model, every time.
