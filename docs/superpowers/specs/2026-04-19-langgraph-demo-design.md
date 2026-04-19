# LangGraph Demo for the Assitio / Midtchat Python Layer — Design

**Date:** 2026-04-19
**Status:** Draft, pending user review
**Author:** Brainstorming session

## 1. Purpose & Scope

Build a fully runnable, fully mocked Python demo that implements the HR Assistant flow from the Assitio / Midtchat Technical Reference on top of [LangGraph](https://langchain-ai.github.io/langgraph/). The goal is **learning LangGraph** using a concrete, realistic flow — not influencing the production direction of Assitio (which has explicitly decided against adopting LangChain as a framework).

The demo is scoped to a single Assistant (HR), with no real external dependencies. Streaming, fan-out retrieval, conditional routing, interrupts, and checkpoint persistence all use LangGraph's native primitives. Assitio-specific concepts (sensitivity enforcement, field-stripping by declared accepts) are layered on top of LangGraph, deliberately, so the boundary between "what the framework gives you" and "what you add for your domain" is clear.

### In scope
- HR Assistant flow with parallel multi-source retrieval
- Mocked LLM and mocked retrieval (no API keys, no vector DB)
- Sensitivity classifier + interrupt gate for clinical-level queries
- Field-stripping decorator implementing the Assitio `accepts` pattern
- In-memory checkpointer for multi-turn conversation persistence
- Jupyter notebook walkthrough + CLI script, sharing the same graph module
- A light `pytest` suite (~10 tests) that proves the key mechanics

### Explicitly out of scope
- Real LLM providers, Weaviate, or any vector store
- MyMidtchat cross-Assistant forwarding and Clinical Guidelines Assistant
- FastAPI / HTTP wrapper (the doc's `POST /api/chat` endpoint)
- LangGraph Studio / Cloud configuration
- CI, Docker, Makefile, coverage thresholds
- Production-grade audit logging, retries, or distributed tracing

### Relationship to Assitio's "no framework" stance
Assitio's architecture explicitly states: *"We do not adopt LangChain as a framework, but we borrow its mental model… The goal is agent flows that are explicit, readable, and debuggable by the team — not abstracted away by a framework."* This demo is a learning exercise, not a counter-proposal. If later it becomes useful as an apples-to-apples comparison artifact, it can be promoted to Approach 1 (Evaluation) — but that is not its current purpose.

## 2. Approach Selected

Among three considered approaches (faithful linear port, LangGraph-idiomatic parallel fan-out, subgraph composition), the design uses **parallel fan-out**: the Router selects 1-3 Feed Agents dynamically and LangGraph's `Send` API drives them concurrently. A fan-in reducer merges their results before the LLM node runs. This is the shortest path that exercises every LangGraph capability targeted by the learning goals.

Capabilities exercised:

| Capability | Where in the graph |
|---|---|
| Typed state + reducers | `HRState` TypedDict with `Annotated[list, add]` on `retrieved_chunks` and `routing_path` |
| Conditional edges | `router` → dynamic `Send` to a subset of feed agents |
| Parallel branching + fan-in | Feed agents run concurrently; reducer merges their writes |
| Interrupts | `interrupt_gate` pauses for human approval when the classifier elevates sensitivity |
| Persistence / checkpointing | `MemorySaver` checkpointer keyed by `thread_id` |
| Streaming | `llm_answer` streams tokens via `astream_events` |

## 3. Graph Shape

```
                     ┌──────────────────────┐
                     │ classify_sensitivity │  (entry — regex-level classifier)
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │   interrupt_gate     │  ←── interrupts if flagged 'clinical'
                     └──────────┬───────────┘         (resumes on human approve)
                                │
                     ┌──────────▼───────────┐
                     │       rewriter       │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │        router        │  (conditional edge below)
                     └──────────┬───────────┘
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼                (parallel fan-out)
         ┌────────────┐ ┌─────────────┐ ┌───────────────────┐
         │feed_policy │ │feed_benefits│ │feed_collective_agr│
         └──────┬─────┘ └──────┬──────┘ └─────────┬─────────┘
                └─────────────┬┴──────────────────┘            (fan-in)
                              ▼
                     ┌────────────────┐
                     │ merge_and_rank │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │   llm_answer   │  (streams tokens)
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ extract_memory │  (flags memory candidates)
                     └────────┬───────┘
                              ▼
                              END        (checkpointer saves state per thread_id)
```

The router's conditional edge picks a subset (1-3) of feed agents based on the rewritten query. "Parental leave" routes to `feed_policy` + `feed_collective_agreement`; "gym membership" routes only to `feed_benefits`. The `interrupt_gate` is a no-op for most queries; it only pauses when the classifier has elevated the effective sensitivity above the declared sensitivity.

## 4. State Schema

A single `TypedDict` flows through the graph. Fields are grouped by who writes them; LangGraph reducers handle the fan-in merge.

```python
from typing import TypedDict, Annotated, Literal
from operator import add

Sensitivity = Literal["public", "internal", "sensitive", "clinical"]

class HRState(TypedDict):
    # --- Input (from "Laravel") — set once, never mutated ---
    user_id: int
    assistant_id: str
    message: str                     # raw user query
    system_prompt: str
    declared_sensitivity: Sensitivity
    memory_context: dict             # {personal: [...], department: [...]}
    config: dict                     # {model, verbosity, language}

    # --- Written by classify_sensitivity ---
    effective_sensitivity: Sensitivity
    sensitivity_reason: str | None

    # --- Written by interrupt_gate (after human resume, if any) ---
    human_approval: Literal["approved", "denied", "not_needed"]

    # --- Written by rewriter ---
    rewritten_query: str

    # --- Written by router ---
    selected_feeds: list[str]
    routing_reason: str

    # --- Written IN PARALLEL by each feed agent → reducer concatenates ---
    retrieved_chunks: Annotated[list[dict], add]

    # --- Written by merge_and_rank ---
    ranked_chunks: list[dict]

    # --- Written by llm_answer ---
    response_text: str
    sources: list[dict]

    # --- Written by extract_memory ---
    memory_candidates: list[dict]

    # --- Debug / audit trail — reducer appends ---
    routing_path: Annotated[list[str], add]
```

The `Annotated[list, add]` reducer on `retrieved_chunks` is what makes parallel fan-out safe — each feed agent returns `{"retrieved_chunks": [...its chunks...]}` and the reducer concatenates. The same pattern on `routing_path` yields a clean audit trail.

## 5. Nodes

Each node is an `async` function taking `HRState` and returning a partial state dict. All node bodies are short; the demo's value is in how they connect.

| Node | Accepts (stripped to these) | Returns | Notes |
|---|---|---|---|
| `classify_sensitivity` | `message` | `effective_sensitivity`, `sensitivity_reason` | Regex pass for CPR / patient-id patterns. Returns `clinical` if matched, else passes through `declared_sensitivity`. |
| `interrupt_gate` | `effective_sensitivity`, `declared_sensitivity` | `human_approval` | If `effective > declared`, calls `interrupt(...)`. Resumes with `Command(resume={"approved": True/False})`. Otherwise `human_approval = "not_needed"`. |
| `rewriter` | `message`, `config.language` | `rewritten_query`, `routing_path: ["rewriter"]` | Mock: strips filler words, normalizes casing. |
| `router` | `rewritten_query` | `selected_feeds`, `routing_reason`, `routing_path: ["router"]` | Keyword-based pick of 1-3 feeds. Returns `Send` objects for parallel fan-out. |
| `feed_policy`, `feed_benefits`, `feed_collective_agreement` | `rewritten_query`, `config.language` | `retrieved_chunks: [...]`, `routing_path: [name]` | Each has its own small hardcoded corpus (~5 chunks). Returns top-3 by keyword score. |
| `merge_and_rank` | `retrieved_chunks` | `ranked_chunks` | Dedupe by `chunk_id`, sort by score, take top 5. |
| `llm_answer` | `rewritten_query`, `ranked_chunks`, `system_prompt`, `memory_context`, `config` | `response_text`, `sources` | Mock LLM composes a templated response referencing top chunks, yields tokens in `astream` mode with a ~20ms sleep per token so streaming is visible. |
| `extract_memory` | `message`, `response_text` | `memory_candidates` | Pattern-matches "I work in X" / "my team is Y" etc. Returns empty list by default. |

The `router → feed_*` parallel edge uses LangGraph's `Send` API. `router` returns something like `[Send("feed_policy", state), Send("feed_benefits", state)]`.

The `interrupt_gate` does not reject clinical queries outright; it pauses so a human can inspect and resume. In production this would be a moderator; in the demo it's `input()` in the notebook / CLI.

## 6. Data Flow — Two Concrete Runs

### Run A: "What are the rules for parental leave?" (normal path)

1. CLI invokes `graph.astream(input_state, config={"configurable": {"thread_id": "demo-1"}})`.
2. `classify_sensitivity` — regex sees nothing suspicious → `effective_sensitivity = "internal"`.
3. `interrupt_gate` — `effective == declared` → `human_approval = "not_needed"`, passes through.
4. `rewriter` — `"what are the rules for parental leave"` → `"parental leave rules"`.
5. `router` — keyword match on `parental|leave` → `selected_feeds = ["feed_policy", "feed_collective_agreement"]`. Emits two `Send` objects.
6. **Parallel:** `feed_policy` returns 3 chunks, `feed_collective_agreement` returns 3 chunks — reducer concatenates into `retrieved_chunks` (6 items).
7. `merge_and_rank` — dedupes, sorts by score, keeps top 5.
8. `llm_answer` — streams tokens via `astream_events`. CLI prints as they arrive.
9. `extract_memory` — no match → empty list.
10. END. Checkpointer persists the full state under `thread_id="demo-1"`.

### Run B: "My patient with CPR 0102031234 needs leave — what applies?" (interrupt path)

1. Same entry.
2. `classify_sensitivity` — CPR regex matches → `effective_sensitivity = "clinical"`, `sensitivity_reason = "CPR pattern detected"`.
3. `interrupt_gate` — `effective ("clinical") > declared ("internal")` → calls `interrupt({"reason": ..., "query": ...})`. Graph pauses.
4. Notebook cell / CLI prompt shows the interrupt payload and asks for approval.
5. User resumes with `graph.ainvoke(Command(resume={"approved": False}), config=...)`.
6. Gate sees `approved=False` → sets `human_approval = "denied"` and short-circuits to END with a refusal message in `response_text`. (If `approved=True`, continues normally. The doc's stance is that clinical-level data should not flow through an `internal` assistant, so denial is the default.)
7. Checkpointer records the whole thing, including the interrupt and resume, for the audit trail.

### Multi-turn

Invoking the graph again with the same `thread_id` replays the checkpoint. The notebook will demonstrate this: run the graph, then run again with a follow-up question — memory context from turn 1 is available in turn 2 because the checkpointer reloaded it.

## 7. Sensitivity Enforcement & Error Handling

Three distinct failure kinds, each handled differently so the demo shows what LangGraph gives you vs. what you layer on top.

### Sensitivity violation (Assitio's gate)
- `validate_flow()` is a **pre-invoke check** that runs *before* `graph.ainvoke` / `graph.astream` is called — not as a graph node. It walks the static list of agents that the compiled graph can reach against the `AGENTS` config (sensitivity level + accepts). If any agent accepts sensitive fields but is rated below the assistant's sensitivity, it raises `SensitivityViolation` and the graph is never invoked. Mirrors the doc's "validate the entire intended flow as a unit, before any of it is executed" rule.
- The field-stripping decorator handles per-node enforcement *during* execution. When a node is about to run, the decorator intersects state keys with its `accepts` list and passes only those. Stripped fields get logged: `stripped for feed_benefits: {patient_context}`.

### Interrupt path
Already covered in Run B above. Not an error — a first-class control-flow mechanism.

### Node failures
- Each node wraps its body in `try/except` and, on failure, logs a structured error and returns `{"routing_path": ["<node>:failed"]}` — the failure is visible in the audit trail without adding a separate state field.
- `merge_and_rank` tolerates empty `retrieved_chunks` and returns `ranked_chunks = []`. The LLM node then composes a graceful "I couldn't find relevant sources" response instead of hallucinating.
- No retry logic — retries belong in a real LLM wrapper; mocking them would teach nothing.

### Logging
One structured logger (`structlog` or stdlib) with fields `{node, thread_id, event}`. Every node logs entry + exit + any stripped fields. The notebook renders a clean audit trail table at the end.

## 8. Testing

Deliberately light. Tests exist to (a) catch regressions while iterating and (b) show patterns for testing LangGraph code.

| Test | What it proves |
|---|---|
| `test_state_reducer_merges_parallel_chunks` | The `Annotated[list, add]` reducer concatenates writes from parallel feed agents. |
| `test_router_selects_expected_feeds` | Given seed queries, `router` emits the expected `Send` targets. |
| `test_field_stripping_decorator` | Decorator hides unauthorized fields from the wrapped node. |
| `test_sensitivity_classifier_regex` | CPR + patient-id patterns get flagged `clinical`; clean queries pass. |
| `test_graph_end_to_end_normal` | Run the full graph on "parental leave" query, assert terminal state. |
| `test_graph_interrupt_and_resume` | Run on a CPR-containing query, assert interrupt, resume with denial, assert terminal state. |
| `test_checkpointer_persists_thread` | Run twice with same `thread_id`, assert second invocation sees first-turn state. |

Framework: `pytest` + `pytest-asyncio`. Target: under 2 seconds total.

Not testing: individual mock outputs, every exception path, performance.

## 9. Project Layout

```
midt_demo_assistant/
├── Assitio_Architecture_Overview_v2.docx          (existing)
├── Assitio_Technical_Reference.docx               (existing)
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-19-langgraph-demo-design.md   (this file)
├── pyproject.toml
├── README.md
├── src/
│   └── hr_assistant/
│       ├── __init__.py
│       ├── state.py              # HRState TypedDict + reducers
│       ├── config.py             # AGENTS dict, SENSITIVITY_LEVELS, SENSITIVE_FIELDS
│       ├── guards.py             # validate_flow(), accepts-decorator, classify_sensitivity()
│       ├── nodes.py              # all node functions (async)
│       ├── graph.py              # StateGraph builder, compile, checkpointer wiring
│       ├── mocks.py              # MockLLM, MockRetriever, seed corpus
│       └── logging.py            # structlog setup
├── tests/
│   ├── test_reducer.py
│   ├── test_router.py
│   ├── test_guards.py
│   ├── test_classifier.py
│   ├── test_graph_e2e.py
│   └── test_checkpointer.py
├── demo_cli.py                   # python demo_cli.py "question?"
└── notebooks/
    └── hr_assistant_walkthrough.ipynb
```

`src/hr_assistant/` keeps the importable demo code tidy and lets the notebook + CLI share exactly the same modules. Files are kept under ~150 lines each. `guards.py` groups the Assitio flavor in one place, so it can be deleted if a pure-LangGraph comparison is ever wanted. `mocks.py` is the one place to swap if a real-LLM version is attempted later.

### Notebook structure
1. Assitio + LangGraph mapping
2. Graph visualization (Mermaid render)
3. State schema walkthrough
4. Normal scenario with per-node state deltas
5. Interrupt scenario with resume
6. Multi-turn with checkpointer
7. Audit table drawn from structured logs

## 10. Dependencies

- Python 3.11+
- `langgraph` (latest stable)
- `pydantic` (for any non-TypedDict value objects, optional)
- `structlog` for logging
- `pytest`, `pytest-asyncio` for tests
- `jupyter` / `ipykernel` for the notebook

Managed via `pyproject.toml`. No external services.

## 11. Success Criteria

1. `python demo_cli.py "What are the rules for parental leave?"` streams a response to stdout and exits cleanly.
2. `python demo_cli.py "My patient with CPR 0102031234 needs leave"` pauses, prints the interrupt payload, accepts `y/n` at a prompt, and completes accordingly.
3. The notebook renders the graph as a Mermaid diagram, runs both scenarios, shows per-node state deltas, and ends with an audit-trail table.
4. `pytest` passes in under 2 seconds.
5. A reader new to the codebase can open `graph.py` and understand the overall flow in under 5 minutes.

## 12. Open Questions

None at the time of writing. The design has been walked through section by section with the user during brainstorming and each section was approved.
