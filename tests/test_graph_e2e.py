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


@pytest.mark.asyncio
async def test_field_stripping_is_wired_for_every_node(capsys):
    """Runtime enforcement: graph invocation must log `fields_stripped` for every node
    that has fields in the input state beyond its declared `accepts`."""
    import re as _re
    from hr_assistant.logging import configure_logging
    configure_logging()
    graph = build_graph()
    state = default_input("Parental leave policy?")
    await graph.ainvoke(state, config={"configurable": {"thread_id": "t-strip"}})

    captured = capsys.readouterr()
    raw = captured.out + captured.err
    # structlog ConsoleRenderer wraps output in ANSI color codes; strip before matching.
    output = _re.sub(r"\x1b\[[0-9;]*m", "", raw)
    # One log line per stripped node; we expect most nodes to strip something.
    assert output.count("fields_stripped") >= 5, (
        f"expected multiple fields_stripped events, got {output.count('fields_stripped')}:\n{output}"
    )
    # Spot-check specific nodes we know have extra state fields to strip.
    for node in ("router", "llm_answer", "feed_policy"):
        assert f"agent={node}" in output, output


@pytest.mark.asyncio
async def test_routing_reason_redacts_tokens_under_elevated_sensitivity():
    """When effective_sensitivity is elevated, the router must not echo query tokens
    into routing_reason."""
    graph = build_graph()
    # CPR in the query will elevate effective_sensitivity to 'clinical'.
    state = default_input("Patient CPR 0102031234 parental leave rules?")
    cfg = {"configurable": {"thread_id": "t-redact"}}
    await graph.ainvoke(state, config=cfg)
    # Approve so the router actually runs
    resumed = await graph.ainvoke(Command(resume={"approved": True}), config=cfg)
    reason = resumed.get("routing_reason", "")
    assert "parental" not in reason.lower()
    assert "leave" not in reason.lower()
    assert "redacted" in reason.lower() or "sensitivity" in reason.lower()
