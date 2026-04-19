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
