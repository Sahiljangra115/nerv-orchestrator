# NERV

*Local LLM Agent Orchestrator. Named after NERV (Evangelion): one command center
that directs each specialized unit to the job it is best at.*

Take one query, split it into parts, send each part to whichever local model is
best at it, then stitch the answers back together. Runs on local models via
[Ollama](https://ollama.com) (text) and `diffusers` (images). No cloud APIs.

> ✅ **v1 complete.** Phases 0–5 implemented and tested. Phase 6 (image/diffusers)
> and Phase 7 (AB-MCTS search via `treequest`) are wired but gated behind optional
> deps — they degrade gracefully without them.
>
> ✅ **v1.1: trained router.** A 22M-param sentence-embedding classifier now does
> question separation and routing in milliseconds, replacing the LLM planner call
> for the common case. See [Trained router](#trained-router).

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
python -m trainer train      # ~30s: builds router.pkl (type head) from trainer/data.yaml
python -m orchestrator "write a Python fibonacci function and explain its big-O"
```

---

## Example trace

Real capture on this machine (trained router, coder `qwen3.5:4b`, general
`gemma4:e2b`, CPU-bound). The clause splitter cut the query at "and explain",
the type head labelled the halves `code` and `explain`, and the router sent
each to its specialist. No LLM call before the work starts:

```
$ python -m orchestrator "Write a Python fibonacci function and explain its big-O."

[planner] learned 2 subtasks (28156ms)     # cold: torch import + MiniLM load. ~7ms warm
[route] #0 code -> coder
[route] #1 explain -> general
[done]  #0 coder (64.9s)
[done]  #1 general (51.1s)
[merge] synthesizing ...

--- final ---
## Python Fibonacci Function and Big-O Analysis
...
def fibonacci_optimized(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
...
#### Time Complexity: O(n) (Linear Time)
The algorithm uses a single for loop that executes approximately n times ...
#### Space Complexity: O(1) (Constant Space)
```

With `--llm-plan` the same query goes through the LLM planner instead
(`[planner] llm 2 subtasks`, ~45s on this CPU before any subtask runs).

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
query -> PLANNER
         learned (default when router.pkl exists): regex clause split ->
           MiniLM embedding -> LogisticRegression type per clause -> merge
           same-type neighbours -> chain depends_on -> optional cost head
           picks the cheapest good-enough model (model_hint). ~10ms warm.
         llm (--llm-plan, or no router.pkl): format=json, Pydantic validate,
           retry x2, fallback single-chat plan, never crashes. ~45s on CPU.
      -> subtasks with depends_on
      -> Kahn topological order:
           ROUTER  -> model_hint from the cost head if set
                      else specialty match within modality
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
| `planner.py`      | LLM fallback: query → validated `Plan` (validate + retry)   |
| `learned.py`      | Trained planner: split → classify → `Plan` with cost hints  |
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
  planner.py           LLM planner (fallback): query -> validated Plan
  learned.py           trained planner + cost-aware router (uses router.pkl)
  router.py            subtask -> model (rule-based + model_hint, deterministic)
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
  test_learned.py      clause splitter, cost objective, hint routing, fake-head plan()
trainer/
  data.yaml            192 labelled questions (32 per type), also the cost pool
  __main__.py          train | collect | eval
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
  --llm-plan        use the LLM planner even when router.pkl exists
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

## Trained router

The LLM planner costs one slow call (~45s on this CPU box) before any work
starts, and it routes purely by subtask *type*. The trained router replaces it
with a small model that answers two questions per clause: **what kind of task
is this** and **which local model answers it well enough for the least cost**.

```bash
python -m trainer train                 # type head from trainer/data.yaml -> router.pkl
python -m trainer eval                  # held-out accuracy on bench/tasks.yaml + split demo
python -m trainer collect --limit 20    # (slow, Ollama) label cost/quality per model
python -m trainer train                 # now also fits the cost head
```

**How it works**

1. **Separation.** A regex clause splitter breaks the query on sentence ends,
   ", then" / ", also", and "and <imperative>" ("write X **and explain** Y").
   Each clause is classified; neighbours with the same type merge back. Each
   subtask depends on the previous one so "explain its big-O" sees the code.
2. **Type head.** `all-MiniLM-L6-v2` sentence embeddings (22M params, CPU) →
   LogisticRegression over six types. Trained on 192 hand-written questions ×
   3 prefix variants. Confidence below 0.35 falls back to `chat` (default model).
3. **Cost head** (optional). `trainer collect` runs *every* text model in
   `models.yaml` on every pool question, scores each answer with the same judge
   the executor uses, and records Ollama's timing minus model-load time. The
   label is the **cheapest model whose score is within 0.1 of the best**. A
   second classifier learns query → model name and the router honours it as
   `model_hint`. This is the RouteLLM idea applied to your own local models:
   never pay for the big model when the small one is about as good.

**Numbers (this machine, `trainer eval`)**

| Metric                                   | Value              |
|------------------------------------------|--------------------|
| Type head, grouped 5-fold CV (MiniLM)    | 0.89               |
| Type head, grouped 5-fold CV (TF-IDF)    | 0.76 (not used)    |
| Held-out accuracy on `bench/tasks.yaml`  | 18/20 = 0.90       |
| `plan()` latency, warm                   | ~7 ms              |
| First call (torch import + model load)   | ~15 s              |
| LLM planner call it replaces             | ~45 s              |

**Honest limits**

- 192 seed questions is small. It generalises because the embeddings are
  pre-trained, not because the dataset is large. Add rows to `trainer/data.yaml`
  whenever the router misfires on something you typed, then retrain.
- The splitter is a regex. "Write X and explain Y" splits; a two-part ask with
  no verb cue after "and" does not. Use `--llm-plan` for real DAGs.
- Cost labels come from an LLM judge that is also one of the candidates
  (`gemma4:e2b` judges itself). Expect self-preference bias; the score gate
  and the 0.1 margin blunt it, they do not remove it. With two local models the
  cost head is a binary coder-vs-general choice; it becomes interesting once
  `models.yaml` has three or more text models.
- Collection is slow: 2 generations + 2 judge calls per question, ~4-6 min per
  question cold on this hardware. It is resumable; run it overnight in chunks.

## Benchmark harness

```bash
ollama serve   # in another terminal
python -m bench run        # runs 20 tasks × 3 judge repeats, saves results/
python -m bench report     # prints win-rate table by subtask type
python -m bench run --planner llm   # force the LLM planner (default: learned if router.pkl)
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
