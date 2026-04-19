from __future__ import annotations

SENSITIVITY_LEVELS = ["public", "internal", "sensitive", "clinical"]

SENSITIVE_FIELDS = ["patient_context", "clinical_notes", "cpr_number"]

AGENTS: dict[str, dict] = {
    "classify_sensitivity": {
        "sensitivity": "public",
        "accepts": ["message", "declared_sensitivity"],
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
        "accepts": ["rewritten_query", "effective_sensitivity"],
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
