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
