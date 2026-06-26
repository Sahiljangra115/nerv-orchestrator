# NERV (orchestrator)

> 🚧 **Under process / WIP.** Scaffold stage. Logic is stubbed (`# TODO(gemini)`); not runnable end-to-end yet.

*Local LLM Agent Orchestrator. Named after NERV (Evangelion): one command center that directs each specialized unit to the job it is best at.*

Take one query, split it into parts, send each part to whichever local model is
best at it, then stitch the answers back together. Runs on local models via
[Ollama](https://ollama.com) (text) and `diffusers` (images). No cloud APIs.

This repo is a **scaffold**: every file exists with a precise contract, but the
logic is left as `# TODO(gemini): ...` stubs to be implemented. See `HANDOFF.md`
for the build prompt.

## What it is (and is not)

- **Is:** decompose + route + synthesize (the RouteLLM / Mixture-of-Agents idea).
- **Is not (yet):** Sakana AB-MCTS ensemble search. That is the v2 bolt-on in
  Phase 7, wired but inert for v1.

## Architecture

```
query -> PLANNER -> subtasks (with deps) -> for each, in order:
                                              ROUTER  -> pick model by type
                                              EXECUTOR-> run on Ollama / SD, feed deps
         SYNTHESIZER <- collect results <-----┘
              |
              v
        final answer + trace
```

Five parts: **Registry** (who is good at what, `models.yaml`), **Planner**
(split), **Router** (assign), **Executor** (run), **Synthesizer** (merge). One
file, `llm.py`, owns all Ollama I/O.

## Layout

```
models.yaml            registry: model -> {specialties, modality, tag}
orchestrator/
  schema.py            Pydantic contracts (written in full; the shared language)
  llm.py               the only file that talks to Ollama
  registry.py          load models.yaml, lookups
  planner.py           query -> validated Plan (validate + retry)
  router.py            subtask -> model (rule-based)
  executor.py          run subtasks in dependency order
  synthesizer.py       merge results into one answer
  cli.py               entrypoint, prints the trace
tests/test_router.py   deterministic routing spec
```

## Build order

0. `llm.py` - talk to one model.
1. `registry.py` + `router.py` - make `tests/test_router.py` pass.
2. `planner.py` - query -> valid JSON Plan (start with one subtask, then many).
3. `executor.py` - run in dependency order, pass outputs forward.
4. `synthesizer.py` - merge.
5. `cli.py` - wire it all, print the trace.
6. image path (`run_image` in `executor.py`, `diffusers`).
7. v2 search (`solve_hard` in `executor.py`, `treequest`).

## Run

```bash
pip install -r requirements.txt
ollama serve            # in another terminal
ollama pull gemma4:e2b  # and whatever models models.yaml lists
python -m pytest tests/
python -m orchestrator "write a python fibonacci function and explain its big-O"
```

Edit `models.yaml` to match the models you actually have pulled.
