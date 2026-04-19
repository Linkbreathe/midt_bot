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
