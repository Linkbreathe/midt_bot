# HR Assistant — LangGraph Demo

A learning-focused LangGraph port of the Assitio / Midtchat HR Assistant flow. Everything external is mocked; the demo runs offline with no API keys.

See the design document at `docs/superpowers/specs/2026-04-19-langgraph-demo-design.md` for the full rationale. The implementation plan is at `docs/superpowers/plans/2026-04-19-langgraph-demo.md`.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### On WSL with the repo on a `/mnt/*` DrvFS path

Windows drives mounted as `/mnt/c`, `/mnt/d`, etc. do not support `chmod`, so the
usual `python -m venv` / editable-install pattern fails mid-way. If `pip install -e`
errors with permission or `chmod` complaints, use this workaround:

```bash
# 1. Create the venv on a POSIX filesystem (your home dir)
python3 -m venv ~/midt_demo_venv

# 2. Symlink it into the repo so everything else "just works"
ln -s ~/midt_demo_venv /mnt/d/Code-Space/midt_demo_assistant/.venv

# 3. Install dependencies directly, then register the source dir via a .pth file
.venv/bin/pip install langgraph structlog pytest pytest-asyncio jupyter ipykernel
echo "/mnt/d/Code-Space/midt_demo_assistant/src" \
  > ~/midt_demo_venv/lib/python3.13/site-packages/hr_assistant_local.pth
```

You'll see a `PytestCacheWarning` every time you run `pytest` because the cache
dir also can't be chmod'd — it's cosmetic and safe to ignore.

## Run the CLI

```bash
.venv/bin/python demo_cli.py "What are the rules for parental leave?"
```

For a sensitivity-elevated query that triggers the interrupt gate:

```bash
.venv/bin/python demo_cli.py "Patient CPR 0102031234 leave rules?"
```

## Run the notebook

```bash
.venv/bin/jupyter notebook notebooks/hr_assistant_walkthrough.ipynb
```

## Run the tests

```bash
.venv/bin/pytest
```

## What's in here

| Path | Purpose |
|---|---|
| `src/hr_assistant/state.py` | `HRState` TypedDict with list-concat reducers for parallel writes |
| `src/hr_assistant/config.py` | `AGENTS` dict — declared `accepts` and sensitivity per agent |
| `src/hr_assistant/guards.py` | Field-stripping decorator, sensitivity classifier, `validate_flow()` |
| `src/hr_assistant/nodes.py` | All async node functions |
| `src/hr_assistant/graph.py` | `StateGraph` assembly, checkpointer wiring, `default_input()` |
| `src/hr_assistant/mocks.py` | Seed corpus, keyword retriever, streaming mock LLM |
| `demo_cli.py` | CLI runner with interactive interrupt handling |
| `notebooks/hr_assistant_walkthrough.ipynb` | Annotated end-to-end walkthrough |

## LangGraph features demonstrated

- Typed state with `Annotated[list, add]` reducers for fan-in merge
- `Send` objects for dynamic parallel fan-out from the router
- `interrupt()` + `Command(resume=...)` for human-in-the-loop gating
- `MemorySaver` checkpointer keyed by `thread_id`
- Token-level streaming via `astream_events`

## What this demo is not

This demo is **not** a proposal to adopt LangChain / LangGraph in the production Assitio codebase. The Assitio architecture explicitly rejects framework coupling in favour of hand-rolled async Python. This repo exists to build LangGraph fluency against a realistic flow, using the HR Assistant as the vehicle.
