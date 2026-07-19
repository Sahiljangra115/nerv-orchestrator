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

# Pull the two text models (or edit models.yaml to match what you have):
ollama pull gemma4:e2b     # general: planner, synth, chat/explain/reason
ollama pull qwen3.5:4b     # coder: code/debug subtasks

python -m pytest tests/
python -m orchestrator "write a Python fibonacci function and explain its big-O"
```

---

## Example trace

Real capture on this machine (planner/synth `gemma4:e2b`, coder `qwen3.5:4b`,
CPU-bound). The planner split the query into a `debug` subtask (routed to the
coder model) and an `explain` subtask (routed to the general model):

```
$ python -m orchestrator "Debug this Python: def fib(n): return fib(n-1)+fib(n-2). Then explain why it fails."

[planner] 2 subtasks
[route] #0 debug -> coder
[route] #1 explain -> general
[done]  #0 coder (102.7s)
[done]  #1 general (47.2s)
[merge] synthesizing ...

--- final ---
The function `def fib(n): return fib(n-1)+fib(n-2)` fails because it defines a
recursive relationship without base cases, leading to infinite recursion and a
RecursionError (stack overflow).

## 1. Why the Original Code Fails
The Fibonacci sequence needs base cases F(0)=0 and F(1)=1. Without them the
calls stack up indefinitely until Python halts execution.
...
## 2. Corrected and Optimized Solutions
def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    return fib(n-1) + fib(n-2)
...
```

> **Honest note on latency:** orchestration is 3 sequential LLM calls (2 subtasks
> + 1 merge) vs. 1 for a raw single-model prompt. On CPU that is real overhead.
> The routing payoff (a code-specialist model on the code subtask) only wins when
> the specialist is genuinely better at that subtype. Whether it beats a single
> model per task type is exactly what the [benchmark harness](#benchmark-harness)
> is for. Also note the planner is only as good as its model: `gemma4:e2b` will
> sometimes mislabel "write a function" as `chat` instead of `code`, which sends
> it to the general model. Routing is only as sharp as the plan.

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
