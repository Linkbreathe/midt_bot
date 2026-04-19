from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import HRState
from .guards import accepts_from_config, validate_flow
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

_NODE_TABLE = [
    ("classify_sensitivity", classify_sensitivity_node),
    ("interrupt_gate", interrupt_gate_node),
    ("rewriter", rewriter_node),
    ("router", router_node),
    ("feed_policy", feed_policy_node),
    ("feed_benefits", feed_benefits_node),
    ("feed_collective_agreement", feed_collective_agreement_node),
    ("merge_and_rank", merge_and_rank_node),
    ("llm_answer", llm_answer_node),
    ("extract_memory", extract_memory_node),
]

_ALL_NODE_NAMES = [name for name, _ in _NODE_TABLE]


def build_graph(checkpointer=None):
    builder = StateGraph(HRState)
    for name, fn in _NODE_TABLE:
        # Wrap every node with the field-stripping decorator so each node
        # only ever sees the fields it declared in AGENTS[name]["accepts"].
        builder.add_node(name, accepts_from_config(name)(fn))

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
