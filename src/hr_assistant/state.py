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
