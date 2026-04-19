from __future__ import annotations

import re

from langgraph.config import get_stream_writer
from langgraph.types import Send, interrupt

from .config import SENSITIVITY_LEVELS
from .guards import classify_with_reason
from .logging import get_logger
from .mocks import MockLLM, MockRetriever

log = get_logger("nodes")

_FILLER = {
    "what", "are", "the", "is", "a", "an", "for", "of", "to",
    "please", "can", "you",
}

_ROUTING_KEYWORDS: dict[str, set[str]] = {
    "feed_policy": {"parental", "sick", "holiday", "vacation", "policy", "rules", "leave"},
    "feed_benefits": {"gym", "pension", "health", "insurance", "benefit", "reimbursement", "salary", "dental"},
    "feed_collective_agreement": {"parental", "overtime", "notice", "agreement", "collective", "termination"},
}

_WORK_PATTERN = re.compile(
    r"I work in (?:the )?([\w\s]+?)(?: department| team|$|\.|,)",
    re.IGNORECASE,
)
_TEAM_PATTERN = re.compile(
    r"my team is (?:the )?([\w\s]+?)(?:$|\.|,)",
    re.IGNORECASE,
)

_ELEVATED = {"sensitive", "clinical"}
_llm = MockLLM()


async def classify_sensitivity_node(state: dict) -> dict:
    level, reason = classify_with_reason(state["message"])
    declared = state.get("declared_sensitivity", "public")
    eff = (
        level
        if SENSITIVITY_LEVELS.index(level) > SENSITIVITY_LEVELS.index(declared)
        else declared
    )
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


async def rewriter_node(state: dict) -> dict:
    message = state["message"].lower().rstrip(".?!")
    tokens = [t for t in message.split() if t not in _FILLER]
    return {
        "rewritten_query": " ".join(tokens),
        "routing_path": ["rewriter"],
    }


async def router_node(state: dict) -> dict:
    q_tokens = set(state["rewritten_query"].split())
    selected = [feed for feed, kws in _ROUTING_KEYWORDS.items() if q_tokens & kws]

    effective = state.get("effective_sensitivity", "public")
    if not selected:
        selected = ["feed_policy"]
        reason = "no keyword match — defaulting to feed_policy"
    elif effective in _ELEVATED:
        # Elevated sensitivity: don't echo query tokens back into state.
        reason = f"sensitivity={effective}; tokens redacted"
    else:
        all_kws = set().union(*_ROUTING_KEYWORDS.values())
        reason = f"matched on tokens: {sorted(q_tokens & all_kws)}"

    return {
        "selected_feeds": selected,
        "routing_reason": reason,
        "routing_path": ["router"],
    }


def router_dispatch(state: dict) -> list[Send]:
    """Conditional-edge function: turn `selected_feeds` into Send objects."""
    return [Send(feed, state) for feed in state["selected_feeds"]]


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


async def llm_answer_node(state: dict) -> dict:
    sources = state.get("ranked_chunks", [])
    # Emit tokens via LangGraph's stream writer so consumers (CLI) can
    # display them as they arrive. Writer is a no-op when streaming is not active.
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    collected: list[str] = []
    async for tok in _llm.astream(state["rewritten_query"], sources):
        if writer is not None:
            writer({"token": tok})
        collected.append(tok)

    return {
        "response_text": "".join(collected).strip(),
        "sources": sources,
        "routing_path": ["llm_answer"],
    }


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
