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
