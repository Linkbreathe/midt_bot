# LangGraph HR Assistant Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, fully mocked LangGraph demo of the Assitio HR Assistant flow with parallel fan-out retrieval, sensitivity interrupt, and checkpoint persistence — delivered as a shared module powering both a CLI and a Jupyter notebook walkthrough.

**Architecture:** Single `StateGraph` with typed `HRState`. Sensitivity classifier → interrupt gate → rewriter → router (conditional `Send` fan-out) → 3 feed agents (parallel) → merge_and_rank → llm_answer (streaming) → extract_memory. `MemorySaver` checkpointer keyed by `thread_id`. Assitio-flavored field-stripping decorator and pre-invoke `validate_flow()` layered on top.

**Tech Stack:** Python 3.11+, `langgraph`, `structlog`, `pytest`, `pytest-asyncio`, `jupyter`.

**Spec:** `docs/superpowers/specs/2026-04-19-langgraph-demo-design.md`

---

## Phase 1 — Foundation

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/hr_assistant/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

Run: `git init && git branch -m main`
Expected: `Initialized empty Git repository`.

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
.ipynb_checkpoints/
.DS_Store
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "hr-assistant-demo"
version = "0.1.0"
description = "LangGraph learning demo based on Assitio HR Assistant flow"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.60",
    "structlog>=24.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 4: Create empty module init files**

`src/hr_assistant/__init__.py`:
```python
"""LangGraph HR Assistant demo — Assitio flavor."""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Create venv and install**

Run: `python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
Expected: `Successfully installed hr-assistant-demo-0.1.0 …`.

- [ ] **Step 6: Verify pytest runs (no tests yet)**

Run: `.venv/bin/pytest`
Expected: `no tests ran in 0.0Xs` (exit code 5 is fine — treat as success for this step).

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml src/hr_assistant/__init__.py tests/__init__.py
git commit -m "chore: project scaffold"
```

---

### Task 2: State Schema and Reducer

**Files:**
- Create: `src/hr_assistant/state.py`
- Create: `tests/test_reducer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reducer.py`:
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import pytest

from hr_assistant.state import HRState


@pytest.mark.asyncio
async def test_state_reducer_merges_parallel_chunks():
    """Two parallel nodes writing to retrieved_chunks should have their outputs concatenated."""

    async def emitter_a(state: HRState) -> dict:
        return {"retrieved_chunks": [{"source": "a", "chunk_id": "a1", "text": "alpha", "score": 0.9}]}

    async def emitter_b(state: HRState) -> dict:
        return {"retrieved_chunks": [{"source": "b", "chunk_id": "b1", "text": "beta", "score": 0.8}]}

    async def fanout(state: HRState):
        return [Send("a", state), Send("b", state)]

    builder = StateGraph(HRState)
    builder.add_node("a", emitter_a)
    builder.add_node("b", emitter_b)
    builder.add_conditional_edges(START, fanout, ["a", "b"])
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    graph = builder.compile()

    result = await graph.ainvoke({"message": "hi", "retrieved_chunks": []})

    assert len(result["retrieved_chunks"]) == 2
    ids = {c["chunk_id"] for c in result["retrieved_chunks"]}
    assert ids == {"a1", "b1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_reducer.py -v`
Expected: `ModuleNotFoundError: No module named 'hr_assistant.state'`.

- [ ] **Step 3: Write minimal `state.py`**

`src/hr_assistant/state.py`:
```python
from __future__ import annotations
from operator import add
from typing import Annotated, Literal, TypedDict

Sensitivity = Literal["public", "internal", "sensitive", "clinical"]


class HRState(TypedDict, total=False):
    # --- Input (from "Laravel") ---
    user_id: int
    assistant_id: str
    message: str
    system_prompt: str
    declared_sensitivity: Sensitivity
    memory_context: dict
    config: dict

    # --- Written by classify_sensitivity ---
    effective_sensitivity: Sensitivity
    sensitivity_reason: str | None

    # --- Written by interrupt_gate ---
    human_approval: Literal["approved", "denied", "not_needed"]

    # --- Written by rewriter ---
    rewritten_query: str

    # --- Written by router ---
    selected_feeds: list[str]
    routing_reason: str

    # --- Written IN PARALLEL by feed agents ---
    retrieved_chunks: Annotated[list[dict], add]

    # --- Written by merge_and_rank ---
    ranked_chunks: list[dict]

    # --- Written by llm_answer ---
    response_text: str
    sources: list[dict]

    # --- Written by extract_memory ---
    memory_candidates: list[dict]

    # --- Audit trail ---
    routing_path: Annotated[list[str], add]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reducer.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/state.py tests/test_reducer.py
git commit -m "feat(state): HRState schema with list-concat reducers"
```

---

### Task 3: Agent Config Dictionary

**Files:**
- Create: `src/hr_assistant/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from hr_assistant.config import AGENTS, SENSITIVITY_LEVELS, SENSITIVE_FIELDS


def test_sensitivity_levels_are_ordered():
    assert SENSITIVITY_LEVELS == ["public", "internal", "sensitive", "clinical"]


def test_all_agents_have_required_keys():
    required = {"sensitivity", "accepts", "type"}
    for name, spec in AGENTS.items():
        assert required.issubset(spec.keys()), f"{name} missing keys"
        assert spec["sensitivity"] in SENSITIVITY_LEVELS


def test_sensitive_fields_declared():
    assert "patient_context" in SENSITIVE_FIELDS
    assert "cpr_number" in SENSITIVE_FIELDS


def test_hr_feed_agents_exist():
    assert "feed_policy" in AGENTS
    assert "feed_benefits" in AGENTS
    assert "feed_collective_agreement" in AGENTS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `config.py`**

`src/hr_assistant/config.py`:
```python
from __future__ import annotations

SENSITIVITY_LEVELS = ["public", "internal", "sensitive", "clinical"]

SENSITIVE_FIELDS = ["patient_context", "clinical_notes", "cpr_number"]

AGENTS: dict[str, dict] = {
    "classify_sensitivity": {
        "sensitivity": "public",
        "accepts": ["message"],
        "type": "classifier",
    },
    "interrupt_gate": {
        "sensitivity": "public",
        "accepts": ["effective_sensitivity", "declared_sensitivity"],
        "type": "gate",
    },
    "rewriter": {
        "sensitivity": "public",
        "accepts": ["message", "config"],
        "type": "rewriter",
    },
    "router": {
        "sensitivity": "internal",
        "accepts": ["rewritten_query"],
        "type": "router",
    },
    "feed_policy": {
        "sensitivity": "internal",
        "accepts": ["rewritten_query", "config"],
        "type": "feed_agent",
    },
    "feed_benefits": {
        "sensitivity": "internal",
        "accepts": ["rewritten_query", "config"],
        "type": "feed_agent",
    },
    "feed_collective_agreement": {
        "sensitivity": "internal",
        "accepts": ["rewritten_query", "config"],
        "type": "feed_agent",
    },
    "merge_and_rank": {
        "sensitivity": "internal",
        "accepts": ["retrieved_chunks"],
        "type": "reducer",
    },
    "llm_answer": {
        "sensitivity": "internal",
        "accepts": ["rewritten_query", "ranked_chunks", "system_prompt", "memory_context", "config"],
        "type": "llm",
    },
    "extract_memory": {
        "sensitivity": "public",
        "accepts": ["message", "response_text"],
        "type": "extractor",
    },
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/config.py tests/test_config.py
git commit -m "feat(config): AGENTS dict, sensitivity levels, sensitive fields"
```

---

## Phase 2 — Guards

### Task 4: Field-Stripping Decorator

**Files:**
- Create: `src/hr_assistant/logging.py`
- Create: `src/hr_assistant/guards.py`
- Create: `tests/test_guards.py`

- [ ] **Step 1: Write the failing test**

`tests/test_guards.py`:
```python
import pytest

from hr_assistant.guards import accepts_from_config


@pytest.mark.asyncio
async def test_decorator_strips_unauthorized_fields():
    """Decorator should pass only the keys declared in AGENTS[name]['accepts']."""

    received = {}

    @accepts_from_config("rewriter")
    async def fake_node(state):
        received.update(state)
        return {"rewritten_query": state["message"].lower()}

    out = await fake_node({
        "message": "HELLO",
        "declared_sensitivity": "internal",
        "patient_context": {"id": 7},
        "config": {"language": "en"},
    })

    assert set(received.keys()) == {"message", "config"}
    assert "patient_context" not in received
    assert out == {"rewritten_query": "hello"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guards.py::test_decorator_strips_unauthorized_fields -v`
Expected: `ImportError: cannot import name 'accepts_from_config'`.

- [ ] **Step 3: Write `logging.py`**

`src/hr_assistant/logging.py`:
```python
import logging
import structlog


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
```

- [ ] **Step 4: Write the decorator in `guards.py`**

`src/hr_assistant/guards.py`:
```python
from __future__ import annotations
from functools import wraps
from typing import Awaitable, Callable

from .config import AGENTS
from .logging import get_logger

log = get_logger("guards")

NodeFn = Callable[[dict], Awaitable[dict]]


def accepts_from_config(agent_name: str):
    """Wrap an async node so it only sees fields in AGENTS[agent_name]['accepts']."""
    allowed = set(AGENTS[agent_name]["accepts"])

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        async def wrapper(state: dict) -> dict:
            stripped = {k: v for k, v in state.items() if k in allowed}
            dropped = set(state.keys()) - allowed - {"routing_path"}
            if dropped:
                log.info("fields_stripped", agent=agent_name, dropped=sorted(dropped))
            return await fn(stripped)
        return wrapper

    return decorator
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_guards.py -v`
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/hr_assistant/guards.py src/hr_assistant/logging.py tests/test_guards.py
git commit -m "feat(guards): accepts-based field-stripping decorator"
```

---

### Task 5: Sensitivity Classifier

**Files:**
- Modify: `src/hr_assistant/guards.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_classifier.py`:
```python
from hr_assistant.guards import classify_query_sensitivity


def test_clean_query_is_public():
    assert classify_query_sensitivity("What are the rules for parental leave?") == "public"


def test_danish_cpr_elevates_to_clinical():
    # Danish CPR: 6 digits + dash + 4 digits, or 10 straight digits
    assert classify_query_sensitivity("Patient CPR 0102031234 needs info") == "clinical"
    assert classify_query_sensitivity("CPR 010203-1234 in file") == "clinical"


def test_patient_id_elevates_to_clinical():
    assert classify_query_sensitivity("See patient P-4567 chart") == "clinical"


def test_reason_returned_alongside():
    from hr_assistant.guards import classify_with_reason
    level, reason = classify_with_reason("CPR 0102031234")
    assert level == "clinical"
    assert "CPR" in reason or "cpr" in reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_classifier.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Add classifier to `guards.py`**

Append to `src/hr_assistant/guards.py`:
```python
import re

from .state import Sensitivity

_CPR_PATTERN = re.compile(r"\b\d{6}-?\d{4}\b")
_PATIENT_ID_PATTERN = re.compile(r"\bP-\d{3,}\b", re.IGNORECASE)


def classify_with_reason(query: str) -> tuple[Sensitivity, str | None]:
    if _CPR_PATTERN.search(query):
        return "clinical", "CPR pattern detected"
    if _PATIENT_ID_PATTERN.search(query):
        return "clinical", "patient ID pattern detected"
    return "public", None


def classify_query_sensitivity(query: str) -> Sensitivity:
    level, _ = classify_with_reason(query)
    return level
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_classifier.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/guards.py tests/test_classifier.py
git commit -m "feat(guards): regex-based query sensitivity classifier"
```

---

### Task 6: Flow Validator

**Files:**
- Modify: `src/hr_assistant/guards.py`
- Modify: `tests/test_guards.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_guards.py`:
```python
from hr_assistant.guards import validate_flow, SensitivityViolation


def test_validate_flow_passes_for_matching_sensitivity():
    # feed_policy is 'internal', assistant is 'internal' — OK
    validate_flow(["rewriter", "feed_policy", "merge_and_rank"], "internal")


def test_validate_flow_passes_public_below_internal():
    # rewriter is 'public' but doesn't accept sensitive fields — OK
    validate_flow(["rewriter"], "internal")


def test_validate_flow_rejects_sensitive_fields_on_low_agent():
    # A fabricated agent that accepts cpr_number but is rated 'public'
    from hr_assistant.config import AGENTS
    AGENTS["_naughty"] = {
        "sensitivity": "public",
        "accepts": ["cpr_number"],
        "type": "test_only",
    }
    try:
        with pytest.raises(SensitivityViolation):
            validate_flow(["_naughty"], "clinical")
    finally:
        del AGENTS["_naughty"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guards.py::test_validate_flow_passes_for_matching_sensitivity -v`
Expected: `ImportError: cannot import name 'validate_flow'`.

- [ ] **Step 3: Implement validator in `guards.py`**

Append to `src/hr_assistant/guards.py`:
```python
from .config import SENSITIVE_FIELDS, SENSITIVITY_LEVELS


class SensitivityViolation(Exception):
    pass


def validate_flow(agent_names: list[str], assistant_sensitivity: Sensitivity) -> None:
    """Raise SensitivityViolation if any agent in the flow would expose
    sensitive fields at too low a sensitivity level."""
    assistant_level = SENSITIVITY_LEVELS.index(assistant_sensitivity)
    for name in agent_names:
        spec = AGENTS[name]
        agent_level = SENSITIVITY_LEVELS.index(spec["sensitivity"])
        accepts_sensitive = any(f in spec["accepts"] for f in SENSITIVE_FIELDS)
        if accepts_sensitive and agent_level < assistant_level:
            raise SensitivityViolation(
                f"{name} accepts sensitive fields but is rated "
                f"'{spec['sensitivity']}' (< '{assistant_sensitivity}')"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_guards.py -v`
Expected: `4 passed` (1 from Task 4 plus 3 here).

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/guards.py tests/test_guards.py
git commit -m "feat(guards): pre-invoke validate_flow for sensitivity enforcement"
```

---

## Phase 3 — Mocks

### Task 7: Mock LLM and Mock Retriever

**Files:**
- Create: `src/hr_assistant/mocks.py`
- Create: `tests/test_mocks.py`

- [ ] **Step 1: Write failing tests**

`tests/test_mocks.py`:
```python
import pytest

from hr_assistant.mocks import MockRetriever, MockLLM, SEED_CORPUS


def test_retriever_returns_top_n_by_keyword():
    retriever = MockRetriever(scope="feed_policy")
    chunks = retriever.retrieve("parental leave", top_n=3)
    assert len(chunks) <= 3
    assert all("chunk_id" in c for c in chunks)
    assert all(c["source"] == "feed_policy" for c in chunks)


def test_retriever_missing_scope_returns_empty():
    retriever = MockRetriever(scope="feed_nonexistent")
    assert retriever.retrieve("anything") == []


def test_seed_corpus_has_three_scopes():
    assert {"feed_policy", "feed_benefits", "feed_collective_agreement"}.issubset(
        SEED_CORPUS.keys()
    )


@pytest.mark.asyncio
async def test_llm_streams_tokens():
    llm = MockLLM()
    tokens = []
    async for tok in llm.astream("Tell me about leave", sources=[]):
        tokens.append(tok)
    assert len(tokens) > 3
    joined = "".join(tokens)
    assert len(joined) > 0


@pytest.mark.asyncio
async def test_llm_empty_sources_yields_apology():
    llm = MockLLM()
    chunks: list[str] = []
    async for tok in llm.astream("anything", sources=[]):
        chunks.append(tok)
    reply = "".join(chunks)
    assert "couldn't find" in reply.lower() or "no relevant" in reply.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mocks.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `mocks.py`**

`src/hr_assistant/mocks.py`:
```python
from __future__ import annotations
import asyncio
from typing import AsyncIterator

SEED_CORPUS: dict[str, list[dict]] = {
    "feed_policy": [
        {"chunk_id": "pol_parental_1", "text": "Employees are entitled to parental leave under company policy §4.2.", "keywords": ["parental", "leave", "policy"]},
        {"chunk_id": "pol_parental_2", "text": "Parental leave may be taken within 14 months of the child's birth.", "keywords": ["parental", "leave", "birth"]},
        {"chunk_id": "pol_sick_1", "text": "Sick leave policy grants up to 30 paid days per calendar year.", "keywords": ["sick", "leave", "paid"]},
        {"chunk_id": "pol_holidays_1", "text": "Employees accrue 25 days of paid holiday per year.", "keywords": ["holiday", "vacation", "paid"]},
    ],
    "feed_benefits": [
        {"chunk_id": "ben_gym_1", "text": "Company gym membership is partially reimbursed up to DKK 300/month.", "keywords": ["gym", "membership", "reimbursement"]},
        {"chunk_id": "ben_pension_1", "text": "Pension contribution is 12% of gross salary, matched by the employer.", "keywords": ["pension", "retirement", "salary"]},
        {"chunk_id": "ben_health_1", "text": "Supplementary health insurance covers dental and physiotherapy.", "keywords": ["health", "insurance", "dental"]},
    ],
    "feed_collective_agreement": [
        {"chunk_id": "ca_parental_1", "text": "Under the collective agreement, parental leave is extended by 4 weeks beyond statutory.", "keywords": ["parental", "leave", "collective", "agreement"]},
        {"chunk_id": "ca_overtime_1", "text": "Overtime above 37 hours/week is compensated at 150% rate.", "keywords": ["overtime", "compensation", "collective"]},
        {"chunk_id": "ca_notice_1", "text": "Notice period is 3 months for employees over 5 years of tenure.", "keywords": ["notice", "termination", "collective"]},
    ],
}


class MockRetriever:
    """Keyword-score retrieval against SEED_CORPUS."""

    def __init__(self, scope: str):
        self.scope = scope

    def retrieve(self, query: str, top_n: int = 3) -> list[dict]:
        corpus = SEED_CORPUS.get(self.scope, [])
        q_tokens = {t.lower() for t in query.split()}
        scored = []
        for chunk in corpus:
            score = sum(1 for kw in chunk["keywords"] if kw.lower() in q_tokens)
            if score > 0:
                scored.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": score / max(len(chunk["keywords"]), 1),
                    "source": self.scope,
                })
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_n]


class MockLLM:
    """Templated response, streamed token-by-token with a short sleep."""

    def __init__(self, token_delay_ms: int = 20):
        self.token_delay_ms = token_delay_ms

    async def astream(self, prompt: str, sources: list[dict]) -> AsyncIterator[str]:
        if not sources:
            text = "I couldn't find relevant sources in the HR knowledge base for your question."
        else:
            snippets = "; ".join(s["text"] for s in sources[:3])
            text = f"Based on the HR knowledge base: {snippets}"
        for token in text.split(" "):
            await asyncio.sleep(self.token_delay_ms / 1000)
            yield token + " "
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_mocks.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/mocks.py tests/test_mocks.py
git commit -m "feat(mocks): seed corpus, keyword retriever, streaming mock LLM"
```

---

## Phase 4 — Nodes

### Task 8: Sensitivity Classifier and Interrupt Gate Nodes

**Files:**
- Create: `src/hr_assistant/nodes.py`
- Create: `tests/test_nodes_gate.py`

- [ ] **Step 1: Write failing tests**

`tests/test_nodes_gate.py`:
```python
import pytest

from hr_assistant.nodes import classify_sensitivity_node, interrupt_gate_node


@pytest.mark.asyncio
async def test_clean_query_keeps_declared_sensitivity():
    state = {"message": "What is our sick leave policy?", "declared_sensitivity": "internal"}
    out = await classify_sensitivity_node(state)
    assert out["effective_sensitivity"] == "internal"
    assert out["sensitivity_reason"] is None


@pytest.mark.asyncio
async def test_cpr_elevates_to_clinical():
    state = {"message": "Patient CPR 0102031234 leave info?", "declared_sensitivity": "internal"}
    out = await classify_sensitivity_node(state)
    assert out["effective_sensitivity"] == "clinical"
    assert "CPR" in (out["sensitivity_reason"] or "")


@pytest.mark.asyncio
async def test_gate_no_op_when_effective_equals_declared():
    state = {"effective_sensitivity": "internal", "declared_sensitivity": "internal"}
    out = await interrupt_gate_node(state)
    assert out["human_approval"] == "not_needed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nodes_gate.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `nodes.py` with gate and classifier nodes**

`src/hr_assistant/nodes.py`:
```python
from __future__ import annotations
from langgraph.types import interrupt

from .config import SENSITIVITY_LEVELS
from .guards import classify_with_reason
from .logging import get_logger

log = get_logger("nodes")


async def classify_sensitivity_node(state: dict) -> dict:
    level, reason = classify_with_reason(state["message"])
    # Effective = max(declared, classified)
    declared = state.get("declared_sensitivity", "public")
    eff = level if SENSITIVITY_LEVELS.index(level) > SENSITIVITY_LEVELS.index(declared) else declared
    log.info("classify_sensitivity", declared=declared, classified=level, effective=eff)
    return {
        "effective_sensitivity": eff,
        "sensitivity_reason": reason,
        "routing_path": ["classify_sensitivity"],
    }


async def interrupt_gate_node(state: dict) -> dict:
    effective = state["effective_sensitivity"]
    declared = state["declared_sensitivity"]
    if SENSITIVITY_LEVELS.index(effective) <= SENSITIVITY_LEVELS.index(declared):
        return {"human_approval": "not_needed", "routing_path": ["interrupt_gate"]}

    decision = interrupt({
        "reason": "classified sensitivity exceeds declared",
        "effective": effective,
        "declared": declared,
    })
    approved = bool(decision.get("approved", False)) if isinstance(decision, dict) else False
    return {
        "human_approval": "approved" if approved else "denied",
        "routing_path": ["interrupt_gate"],
        **({"response_text": "Request denied: sensitivity elevated."} if not approved else {}),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nodes_gate.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/nodes.py tests/test_nodes_gate.py
git commit -m "feat(nodes): classify_sensitivity and interrupt_gate nodes"
```

---

### Task 9: Rewriter Node

**Files:**
- Modify: `src/hr_assistant/nodes.py`
- Create: `tests/test_nodes_rewriter.py`

- [ ] **Step 1: Write failing test**

`tests/test_nodes_rewriter.py`:
```python
import pytest

from hr_assistant.nodes import rewriter_node


@pytest.mark.asyncio
async def test_rewriter_lowers_and_strips_filler():
    out = await rewriter_node({"message": "What are the rules for parental leave?"})
    assert "rewritten_query" in out
    assert out["rewritten_query"].islower()
    assert "what" not in out["rewritten_query"].split()
    assert "rules" in out["rewritten_query"] or "parental" in out["rewritten_query"]


@pytest.mark.asyncio
async def test_rewriter_adds_path_entry():
    out = await rewriter_node({"message": "Sick leave policy"})
    assert out["routing_path"] == ["rewriter"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nodes_rewriter.py -v`
Expected: `ImportError: cannot import name 'rewriter_node'`.

- [ ] **Step 3: Append to `nodes.py`**

```python
_FILLER = {"what", "are", "the", "is", "a", "an", "for", "of", "to", "please", "can", "you"}


async def rewriter_node(state: dict) -> dict:
    message = state["message"].lower().rstrip(".?!")
    tokens = [t for t in message.split() if t not in _FILLER]
    return {
        "rewritten_query": " ".join(tokens),
        "routing_path": ["rewriter"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nodes_rewriter.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/nodes.py tests/test_nodes_rewriter.py
git commit -m "feat(nodes): rewriter node (lowercase, filler-strip)"
```

---

### Task 10: Router Node with Parallel Send

**Files:**
- Modify: `src/hr_assistant/nodes.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write failing test**

`tests/test_router.py`:
```python
import pytest
from langgraph.types import Send

from hr_assistant.nodes import router_node, router_dispatch


@pytest.mark.asyncio
async def test_router_picks_policy_and_ca_for_parental():
    out = await router_node({"rewritten_query": "parental leave rules"})
    assert set(out["selected_feeds"]) == {"feed_policy", "feed_collective_agreement"}
    assert out["routing_path"] == ["router"]


@pytest.mark.asyncio
async def test_router_picks_benefits_only_for_gym():
    out = await router_node({"rewritten_query": "gym membership reimbursement"})
    assert out["selected_feeds"] == ["feed_benefits"]


@pytest.mark.asyncio
async def test_router_falls_back_to_policy_when_no_keyword_match():
    out = await router_node({"rewritten_query": "xyzzy foobar"})
    assert out["selected_feeds"] == ["feed_policy"]


def test_router_dispatch_produces_send_objects():
    state = {"selected_feeds": ["feed_policy", "feed_benefits"], "rewritten_query": "q"}
    sends = router_dispatch(state)
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert {s.node for s in sends} == {"feed_policy", "feed_benefits"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Append to `nodes.py`**

```python
from langgraph.types import Send

_ROUTING_KEYWORDS = {
    "feed_policy": {"parental", "sick", "holiday", "vacation", "policy", "rules", "leave"},
    "feed_benefits": {"gym", "pension", "health", "insurance", "benefit", "reimbursement", "salary", "dental"},
    "feed_collective_agreement": {"parental", "overtime", "notice", "agreement", "collective", "termination"},
}


async def router_node(state: dict) -> dict:
    q_tokens = set(state["rewritten_query"].split())
    selected = [feed for feed, kws in _ROUTING_KEYWORDS.items() if q_tokens & kws]
    if not selected:
        selected = ["feed_policy"]
        reason = "no keyword match — defaulting to feed_policy"
    else:
        reason = f"matched on tokens: {sorted(q_tokens & set().union(*_ROUTING_KEYWORDS.values()))}"
    return {
        "selected_feeds": selected,
        "routing_reason": reason,
        "routing_path": ["router"],
    }


def router_dispatch(state: dict) -> list[Send]:
    """Conditional-edge function: turn `selected_feeds` into Send objects."""
    return [Send(feed, state) for feed in state["selected_feeds"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/nodes.py tests/test_router.py
git commit -m "feat(nodes): router with parallel Send dispatch"
```

---

### Task 11: Feed Agent Nodes

**Files:**
- Modify: `src/hr_assistant/nodes.py`
- Create: `tests/test_nodes_feeds.py`

- [ ] **Step 1: Write failing tests**

`tests/test_nodes_feeds.py`:
```python
import pytest

from hr_assistant.nodes import feed_policy_node, feed_benefits_node, feed_collective_agreement_node


@pytest.mark.asyncio
async def test_feed_policy_returns_policy_scoped_chunks():
    out = await feed_policy_node({"rewritten_query": "parental leave"})
    assert "retrieved_chunks" in out
    assert out["routing_path"] == ["feed_policy"]
    assert all(c["source"] == "feed_policy" for c in out["retrieved_chunks"])


@pytest.mark.asyncio
async def test_feed_benefits_returns_benefits_scoped_chunks():
    out = await feed_benefits_node({"rewritten_query": "gym membership"})
    assert all(c["source"] == "feed_benefits" for c in out["retrieved_chunks"])


@pytest.mark.asyncio
async def test_feed_collective_agreement_empty_ok():
    out = await feed_collective_agreement_node({"rewritten_query": "xyzzy"})
    assert out["retrieved_chunks"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nodes_feeds.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Append to `nodes.py`**

```python
from .mocks import MockRetriever


def _make_feed_node(scope: str):
    retriever = MockRetriever(scope=scope)

    async def feed_node(state: dict) -> dict:
        chunks = retriever.retrieve(state["rewritten_query"], top_n=3)
        return {
            "retrieved_chunks": chunks,
            "routing_path": [scope],
        }

    feed_node.__name__ = f"{scope}_node"
    return feed_node


feed_policy_node = _make_feed_node("feed_policy")
feed_benefits_node = _make_feed_node("feed_benefits")
feed_collective_agreement_node = _make_feed_node("feed_collective_agreement")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nodes_feeds.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/nodes.py tests/test_nodes_feeds.py
git commit -m "feat(nodes): three feed agent nodes using MockRetriever"
```

---

### Task 12: Merge and Rank Node

**Files:**
- Modify: `src/hr_assistant/nodes.py`
- Create: `tests/test_nodes_merge.py`

- [ ] **Step 1: Write failing tests**

`tests/test_nodes_merge.py`:
```python
import pytest

from hr_assistant.nodes import merge_and_rank_node


@pytest.mark.asyncio
async def test_merge_dedupes_by_chunk_id():
    state = {"retrieved_chunks": [
        {"chunk_id": "a", "text": "A", "score": 0.3, "source": "x"},
        {"chunk_id": "a", "text": "A", "score": 0.3, "source": "x"},
        {"chunk_id": "b", "text": "B", "score": 0.9, "source": "y"},
    ]}
    out = await merge_and_rank_node(state)
    assert len(out["ranked_chunks"]) == 2
    assert out["ranked_chunks"][0]["chunk_id"] == "b"  # highest score first


@pytest.mark.asyncio
async def test_merge_caps_at_five():
    state = {"retrieved_chunks": [
        {"chunk_id": f"c{i}", "text": "t", "score": i / 10, "source": "s"}
        for i in range(10)
    ]}
    out = await merge_and_rank_node(state)
    assert len(out["ranked_chunks"]) == 5


@pytest.mark.asyncio
async def test_merge_tolerates_empty():
    out = await merge_and_rank_node({"retrieved_chunks": []})
    assert out["ranked_chunks"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nodes_merge.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Append to `nodes.py`**

```python
async def merge_and_rank_node(state: dict) -> dict:
    seen: set[str] = set()
    deduped: list[dict] = []
    for chunk in state.get("retrieved_chunks", []):
        if chunk["chunk_id"] in seen:
            continue
        seen.add(chunk["chunk_id"])
        deduped.append(chunk)
    deduped.sort(key=lambda c: c["score"], reverse=True)
    return {
        "ranked_chunks": deduped[:5],
        "routing_path": ["merge_and_rank"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nodes_merge.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/nodes.py tests/test_nodes_merge.py
git commit -m "feat(nodes): merge_and_rank with dedupe + top-5 cap"
```

---

### Task 13: LLM Answer Node and Extract Memory Node

**Files:**
- Modify: `src/hr_assistant/nodes.py`
- Create: `tests/test_nodes_llm_memory.py`

- [ ] **Step 1: Write failing tests**

`tests/test_nodes_llm_memory.py`:
```python
import pytest

from hr_assistant.nodes import llm_answer_node, extract_memory_node


@pytest.mark.asyncio
async def test_llm_answer_composes_text_from_sources():
    state = {
        "rewritten_query": "parental leave rules",
        "ranked_chunks": [
            {"chunk_id": "p1", "text": "Parental leave is 24 weeks.", "score": 1.0, "source": "feed_policy"},
        ],
        "system_prompt": "You are HR.",
        "memory_context": {"personal": [], "department": []},
        "config": {"language": "en"},
    }
    out = await llm_answer_node(state)
    assert len(out["response_text"]) > 0
    assert out["sources"][0]["chunk_id"] == "p1"


@pytest.mark.asyncio
async def test_llm_answer_handles_no_sources():
    state = {
        "rewritten_query": "xyzzy",
        "ranked_chunks": [],
        "system_prompt": "",
        "memory_context": {},
        "config": {},
    }
    out = await llm_answer_node(state)
    assert "couldn't find" in out["response_text"].lower()
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_extract_memory_flags_work_statement():
    out = await extract_memory_node({
        "message": "I work in the cardiology department",
        "response_text": "Noted.",
    })
    assert any("cardiology" in c["content"].lower() for c in out["memory_candidates"])
    assert out["memory_candidates"][0]["type"] == "personal"


@pytest.mark.asyncio
async def test_extract_memory_empty_when_no_pattern():
    out = await extract_memory_node({"message": "what time is it", "response_text": "12:00"})
    assert out["memory_candidates"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nodes_llm_memory.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Append to `nodes.py`**

```python
import re as _re
from .mocks import MockLLM

_llm = MockLLM()


async def llm_answer_node(state: dict) -> dict:
    sources = state.get("ranked_chunks", [])
    tokens: list[str] = []
    async for tok in _llm.astream(state["rewritten_query"], sources):
        tokens.append(tok)
    return {
        "response_text": "".join(tokens).strip(),
        "sources": sources,
        "routing_path": ["llm_answer"],
    }


_WORK_PATTERN = _re.compile(r"I work in (?:the )?([\w\s]+?)(?: department| team|$|\.|,)", _re.IGNORECASE)
_TEAM_PATTERN = _re.compile(r"my team is (?:the )?([\w\s]+?)(?:$|\.|,)", _re.IGNORECASE)


async def extract_memory_node(state: dict) -> dict:
    message = state.get("message", "")
    candidates: list[dict] = []
    m = _WORK_PATTERN.search(message)
    if m:
        candidates.append({
            "content": f"User works in the {m.group(1).strip()} department",
            "type": "personal",
        })
    m = _TEAM_PATTERN.search(message)
    if m:
        candidates.append({
            "content": f"User's team is {m.group(1).strip()}",
            "type": "personal",
        })
    return {
        "memory_candidates": candidates,
        "routing_path": ["extract_memory"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nodes_llm_memory.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/nodes.py tests/test_nodes_llm_memory.py
git commit -m "feat(nodes): llm_answer (streaming) and extract_memory nodes"
```

---

## Phase 5 — Graph Assembly

### Task 14: Graph Builder + Normal E2E Test

**Files:**
- Create: `src/hr_assistant/graph.py`
- Create: `tests/test_graph_e2e.py`

- [ ] **Step 1: Write failing test**

`tests/test_graph_e2e.py`:
```python
import pytest

from hr_assistant.graph import build_graph, default_input


@pytest.mark.asyncio
async def test_normal_flow_parental_leave():
    graph = build_graph()
    state = default_input("What are the rules for parental leave?")
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "t-normal"}})

    assert result["human_approval"] == "not_needed"
    assert result["effective_sensitivity"] == "internal"
    assert len(result["response_text"]) > 0
    # Router should have picked policy + collective_agreement
    assert "feed_policy" in result["selected_feeds"]
    assert "feed_collective_agreement" in result["selected_feeds"]
    # Parallel writes concatenated
    assert len(result["retrieved_chunks"]) >= 2
    # Audit trail mentions key nodes
    path = result["routing_path"]
    for expected in ["classify_sensitivity", "interrupt_gate", "rewriter", "router", "merge_and_rank", "llm_answer"]:
        assert expected in path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph_e2e.py -v`
Expected: `ModuleNotFoundError: No module named 'hr_assistant.graph'`.

- [ ] **Step 3: Write `graph.py`**

`src/hr_assistant/graph.py`:
```python
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import HRState
from .guards import validate_flow
from .nodes import (
    classify_sensitivity_node,
    interrupt_gate_node,
    rewriter_node,
    router_node,
    router_dispatch,
    feed_policy_node,
    feed_benefits_node,
    feed_collective_agreement_node,
    merge_and_rank_node,
    llm_answer_node,
    extract_memory_node,
)

_ALL_NODE_NAMES = [
    "classify_sensitivity", "interrupt_gate", "rewriter", "router",
    "feed_policy", "feed_benefits", "feed_collective_agreement",
    "merge_and_rank", "llm_answer", "extract_memory",
]


def build_graph(checkpointer=None):
    builder = StateGraph(HRState)
    builder.add_node("classify_sensitivity", classify_sensitivity_node)
    builder.add_node("interrupt_gate", interrupt_gate_node)
    builder.add_node("rewriter", rewriter_node)
    builder.add_node("router", router_node)
    builder.add_node("feed_policy", feed_policy_node)
    builder.add_node("feed_benefits", feed_benefits_node)
    builder.add_node("feed_collective_agreement", feed_collective_agreement_node)
    builder.add_node("merge_and_rank", merge_and_rank_node)
    builder.add_node("llm_answer", llm_answer_node)
    builder.add_node("extract_memory", extract_memory_node)

    builder.add_edge(START, "classify_sensitivity")
    builder.add_edge("classify_sensitivity", "interrupt_gate")

    # If gate denies, short-circuit straight to END
    def gate_router(state: dict) -> str:
        return "rewriter" if state.get("human_approval") != "denied" else END

    builder.add_conditional_edges("interrupt_gate", gate_router, ["rewriter", END])
    builder.add_edge("rewriter", "router")

    # Parallel fan-out via Send
    builder.add_conditional_edges(
        "router",
        router_dispatch,
        ["feed_policy", "feed_benefits", "feed_collective_agreement"],
    )
    builder.add_edge("feed_policy", "merge_and_rank")
    builder.add_edge("feed_benefits", "merge_and_rank")
    builder.add_edge("feed_collective_agreement", "merge_and_rank")
    builder.add_edge("merge_and_rank", "llm_answer")
    builder.add_edge("llm_answer", "extract_memory")
    builder.add_edge("extract_memory", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def default_input(message: str, sensitivity: str = "internal") -> dict:
    """Pre-invoke: validate flow, then return a fully formed input state."""
    validate_flow(_ALL_NODE_NAMES, sensitivity)
    return {
        "user_id": 42,
        "assistant_id": "hr_assistant",
        "message": message,
        "system_prompt": "You are an HR assistant.",
        "declared_sensitivity": sensitivity,
        "memory_context": {"personal": [], "department": []},
        "config": {"model": "mock", "verbosity": "concise", "language": "en"},
        "retrieved_chunks": [],
        "routing_path": [],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_graph_e2e.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/hr_assistant/graph.py tests/test_graph_e2e.py
git commit -m "feat(graph): StateGraph assembly with checkpointer and validator"
```

---

### Task 15: Interrupt + Resume E2E Test

**Files:**
- Modify: `tests/test_graph_e2e.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_graph_e2e.py`:
```python
from langgraph.types import Command


@pytest.mark.asyncio
async def test_cpr_query_interrupts_and_denies():
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "t-clinical"}}
    state = default_input("Patient CPR 0102031234 leave rules?")

    result = await graph.ainvoke(state, config=cfg)
    # When interrupt fires, ainvoke returns a state that contains __interrupt__
    assert "__interrupt__" in result

    # Resume with denial
    resumed = await graph.ainvoke(Command(resume={"approved": False}), config=cfg)
    assert resumed["human_approval"] == "denied"
    assert "denied" in resumed["response_text"].lower()
    # Retrieval and LLM should NOT have run after denial
    assert "llm_answer" not in resumed.get("routing_path", [])


@pytest.mark.asyncio
async def test_cpr_query_resumes_with_approval_continues_flow():
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "t-clinical-approved"}}
    state = default_input("Patient CPR 0102031234 parental leave?")

    await graph.ainvoke(state, config=cfg)
    resumed = await graph.ainvoke(Command(resume={"approved": True}), config=cfg)
    assert resumed["human_approval"] == "approved"
    assert "llm_answer" in resumed["routing_path"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_graph_e2e.py::test_cpr_query_interrupts_and_denies -v`
Expected: FAIL — likely because the current gate node sets `response_text` only on denial after resume, which should already work; or passes immediately. If it fails with a different reason, read the error before continuing.

- [ ] **Step 3: Check and refine interrupt handling if needed**

If Step 2 showed the test fail because the denial short-circuit didn't happen, verify `gate_router` in `graph.py` routes to `END` when `human_approval == "denied"`. No code change usually needed — this wiring already exists.

If Step 2 passed directly (the test was correct the first time), skip to Step 4.

- [ ] **Step 4: Run all graph tests**

Run: `.venv/bin/pytest tests/test_graph_e2e.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_graph_e2e.py
git commit -m "test(graph): interrupt + resume (deny and approve)"
```

---

### Task 16: Checkpointer Persistence Test

**Files:**
- Create: `tests/test_checkpointer.py`

- [ ] **Step 1: Write failing test**

`tests/test_checkpointer.py`:
```python
import pytest
from langgraph.checkpoint.memory import MemorySaver

from hr_assistant.graph import build_graph, default_input


@pytest.mark.asyncio
async def test_same_thread_id_replays_prior_state():
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "t-multi"}}

    # Turn 1
    r1 = await graph.ainvoke(default_input("Parental leave rules?"), config=cfg)
    assert r1["response_text"]

    # Retrieve state snapshot
    snapshot = await graph.aget_state(cfg)
    assert snapshot.values["response_text"] == r1["response_text"]
    # routing_path has accumulated every node
    assert "llm_answer" in snapshot.values["routing_path"]


@pytest.mark.asyncio
async def test_different_thread_ids_are_isolated():
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    r_a = await graph.ainvoke(default_input("Gym membership?"), config={"configurable": {"thread_id": "A"}})
    r_b = await graph.ainvoke(default_input("Parental leave?"), config={"configurable": {"thread_id": "B"}})

    snap_a = await graph.aget_state({"configurable": {"thread_id": "A"}})
    snap_b = await graph.aget_state({"configurable": {"thread_id": "B"}})

    assert snap_a.values["selected_feeds"] != snap_b.values["selected_feeds"]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_checkpointer.py -v`
Expected: `2 passed` (graph + checkpointer already exist from Task 14).

- [ ] **Step 3: Run full suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass, total under ~3s.

- [ ] **Step 4: Commit**

```bash
git add tests/test_checkpointer.py
git commit -m "test(checkpointer): thread_id isolation and state replay"
```

---

## Phase 6 — Demo Deliverables

### Task 17: CLI Script

**Files:**
- Create: `demo_cli.py`

- [ ] **Step 1: Write the CLI script**

`demo_cli.py`:
```python
"""CLI runner for the HR Assistant demo.

Usage:
    python demo_cli.py "What are the rules for parental leave?"
"""
from __future__ import annotations
import asyncio
import sys

from langgraph.types import Command

from hr_assistant.graph import build_graph, default_input
from hr_assistant.logging import configure_logging


async def run(message: str) -> None:
    configure_logging()
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "cli-session"}}
    state = default_input(message)

    print(f"\n>>> {message}\n")
    # Stream LLM tokens for the llm_answer node
    seen_tokens = 0
    async for event in graph.astream_events(state, config=cfg, version="v2"):
        kind = event.get("event")
        if kind == "on_chain_end" and event.get("name") == "llm_answer":
            data = event["data"].get("output", {})
            text = data.get("response_text", "")
            if text and seen_tokens == 0:
                # Mock LLM accumulates inside the node; print the full output here.
                print(text)
                seen_tokens = 1

    snapshot = await graph.aget_state(cfg)
    values = snapshot.values

    if "__interrupt__" in values or values.get("human_approval") == "denied":
        if values.get("human_approval") != "denied":
            interrupt_payload = values.get("__interrupt__")
            print("\n[!] Graph paused — sensitivity elevated.")
            print(f"    Details: {interrupt_payload}")
            answer = input("    Approve? [y/N]: ").strip().lower()
            approved = answer == "y"
            resumed = await graph.ainvoke(Command(resume={"approved": approved}), config=cfg)
            print("\n" + resumed.get("response_text", ""))
            values = resumed
        else:
            print(values.get("response_text", "Denied."))

    print("\n--- Audit trail ---")
    for step in values.get("routing_path", []):
        print(f"  · {step}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python demo_cli.py \"your question\"")
        sys.exit(2)
    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the CLI with a normal query**

Run: `.venv/bin/python demo_cli.py "What are the rules for parental leave?"`
Expected: the response is printed, then an audit trail listing `classify_sensitivity`, `interrupt_gate`, `rewriter`, `router`, `feed_policy`, `feed_collective_agreement`, `merge_and_rank`, `llm_answer`, `extract_memory`.

- [ ] **Step 3: Run the CLI with a clinical query (answer `n` at prompt)**

Run: `.venv/bin/python demo_cli.py "Patient CPR 0102031234 leave info"`
Expected: script pauses, prints interrupt details, waits for input. Type `n` and press Enter. Final output should be the denial message.

- [ ] **Step 4: Commit**

```bash
git add demo_cli.py
git commit -m "feat(cli): interactive demo runner with interrupt handling"
```

---

### Task 18: Jupyter Notebook Walkthrough

**Files:**
- Create: `notebooks/hr_assistant_walkthrough.ipynb`

- [ ] **Step 1: Create the notebook scaffold with `jupytext`-style plain-JSON**

`notebooks/hr_assistant_walkthrough.ipynb`:
```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
    "# HR Assistant — LangGraph Walkthrough\n",
    "\n",
    "This notebook walks through the HR Assistant demo built on LangGraph. It maps 1:1 onto the Assitio architecture from `Assitio_Architecture_Overview_v2.docx`, with all external dependencies mocked.\n",
    "\n",
    "**Sections:**\n",
    "1. Graph visualization\n",
    "2. State schema\n",
    "3. Normal scenario (with per-node state deltas)\n",
    "4. Interrupt scenario\n",
    "5. Multi-turn with checkpointer\n",
    "6. Audit trail table"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Graph visualization"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "from hr_assistant.graph import build_graph\n",
    "from hr_assistant.logging import configure_logging\n",
    "configure_logging()\n",
    "graph = build_graph()\n",
    "from IPython.display import Image\n",
    "Image(graph.get_graph().draw_mermaid_png())"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 2. State schema"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "from hr_assistant.state import HRState\n",
    "import typing\n",
    "for field, annotation in typing.get_type_hints(HRState, include_extras=True).items():\n",
    "    print(f'{field}: {annotation}')"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Normal scenario with per-node deltas"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "from hr_assistant.graph import default_input\n",
    "cfg = {'configurable': {'thread_id': 'nb-normal'}}\n",
    "state = default_input('What are the rules for parental leave?')\n",
    "async for update in graph.astream(state, config=cfg):\n",
    "    for node, delta in update.items():\n",
    "        print(f'=== {node} ===')\n",
    "        for k, v in delta.items():\n",
    "            print(f'  {k}: {v}')"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Interrupt scenario"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "from langgraph.types import Command\n",
    "cfg = {'configurable': {'thread_id': 'nb-interrupt'}}\n",
    "state = default_input('Patient CPR 0102031234 parental leave?')\n",
    "first = await graph.ainvoke(state, config=cfg)\n",
    "print('Interrupt payload:', first.get('__interrupt__'))"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "resumed = await graph.ainvoke(Command(resume={'approved': False}), config=cfg)\n",
    "print('Final response:', resumed.get('response_text'))\n",
    "print('Approval:', resumed.get('human_approval'))"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Multi-turn with checkpointer"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "cfg = {'configurable': {'thread_id': 'nb-multi'}}\n",
    "await graph.ainvoke(default_input('Parental leave rules?'), config=cfg)\n",
    "snap = await graph.aget_state(cfg)\n",
    "print('Turn 1 routing path:', snap.values['routing_path'])"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Audit trail table"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
    "snap = await graph.aget_state({'configurable': {'thread_id': 'nb-normal'}})\n",
    "from itertools import groupby\n",
    "print('Step  | Node')\n",
    "print('------+------------------------------')\n",
    "for i, step in enumerate(snap.values.get('routing_path', []), 1):\n",
    "    print(f'  {i:>2}  | {step}')"
  ]}
 ],
 "metadata": {
   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
   "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Smoke-test the notebook end-to-end**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute notebooks/hr_assistant_walkthrough.ipynb --output hr_assistant_walkthrough.executed.ipynb`
Expected: no exceptions; `hr_assistant_walkthrough.executed.ipynb` is produced.

- [ ] **Step 3: Delete the executed copy (it's an artifact)**

Run: `rm notebooks/hr_assistant_walkthrough.executed.ipynb`
Expected: file removed.

- [ ] **Step 4: Commit**

```bash
git add notebooks/hr_assistant_walkthrough.ipynb
git commit -m "docs(notebook): end-to-end walkthrough with graph viz and interrupt demo"
```

---

### Task 19: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
# HR Assistant — LangGraph Demo

A learning-focused LangGraph port of the Assitio / Midtchat HR Assistant flow. Everything external is mocked; the demo runs offline with no API keys.

See the design document at `docs/superpowers/specs/2026-04-19-langgraph-demo-design.md` for the full rationale. The implementation plan is at `docs/superpowers/plans/2026-04-19-langgraph-demo.md`.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, usage, and feature overview"
```

---

## Self-Review Notes

Ran a final pass against the spec:

- **Spec section 1 (purpose/scope):** ✓ covered by Task 1 scaffold + README.
- **Spec section 2 (approach):** ✓ parallel fan-out is implemented in Tasks 10-12.
- **Spec section 3 (graph shape):** ✓ Task 14 assembles the exact graph.
- **Spec section 4 (state schema):** ✓ Task 2.
- **Spec section 5 (nodes):** ✓ Tasks 8-13 cover all nine nodes with their declared `accepts`.
- **Spec section 6 (data flow):** ✓ Tasks 14, 15, 16 exercise normal, interrupt, and multi-turn paths end-to-end.
- **Spec section 7 (sensitivity + error handling):** ✓ Task 6 covers `validate_flow`; Task 4 covers field-stripping; Task 15 covers interrupts.
- **Spec section 8 (testing):** ✓ all seven listed test categories are present.
- **Spec section 9 (project layout):** ✓ file tree matches.
- **Spec section 10 (dependencies):** ✓ Task 1 pyproject matches.
- **Spec section 11 (success criteria):** ✓ Tasks 17 (criteria 1, 2), 18 (3), 16 (4), and the overall structure (5).

No placeholders or TBDs remain. Type consistency verified: `HRState` field names used across tasks match Task 2; node function names match across tasks (`feed_policy_node`, `router_dispatch`, etc.).
