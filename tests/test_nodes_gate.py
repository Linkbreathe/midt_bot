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
