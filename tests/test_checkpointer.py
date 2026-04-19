import pytest
from langgraph.checkpoint.memory import MemorySaver

from hr_assistant.graph import build_graph, default_input


@pytest.mark.asyncio
async def test_same_thread_id_replays_prior_state():
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "t-multi"}}

    # Turn 1
    r1 = await graph.ainvoke(default_input("Parental leave rules?"), config=cfg)
    assert r1["response_text"]

    # Retrieve state snapshot
    snapshot = await graph.aget_state(cfg)
    assert snapshot.values["response_text"] == r1["response_text"]
    # routing_path has accumulated every node
    assert "llm_answer" in snapshot.values["routing_path"]


@pytest.mark.asyncio
async def test_different_thread_ids_are_isolated():
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    r_a = await graph.ainvoke(default_input("Gym membership?"), config={"configurable": {"thread_id": "A"}})
    r_b = await graph.ainvoke(default_input("Parental leave?"), config={"configurable": {"thread_id": "B"}})

    snap_a = await graph.aget_state({"configurable": {"thread_id": "A"}})
    snap_b = await graph.aget_state({"configurable": {"thread_id": "B"}})

    assert snap_a.values["selected_feeds"] != snap_b.values["selected_feeds"]


@pytest.mark.asyncio
async def test_multi_turn_state_history_retains_both_turns():
    """Invoking twice on the same thread_id keeps both turns visible in history
    — proving the checkpointer is actually replaying state across invocations."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "t-multiturn"}}

    r1 = await graph.ainvoke(default_input("Parental leave rules?"), config=cfg)
    assert r1["response_text"]
    turn1_message = r1["message"]

    r2 = await graph.ainvoke(default_input("Gym membership reimbursement?"), config=cfg)
    assert r2["response_text"]
    assert r2["message"] == "Gym membership reimbursement?"

    # Walk the full state history — both turns' messages must be present.
    history = [snap async for snap in graph.aget_state_history(cfg)]
    assert len(history) > 2, f"expected multiple snapshots across turns, got {len(history)}"
    messages_in_history = {snap.values.get("message") for snap in history if snap.values.get("message")}
    assert turn1_message in messages_in_history
    assert "Gym membership reimbursement?" in messages_in_history


@pytest.mark.asyncio
async def test_multi_turn_routing_path_reducer_accumulates_across_turns():
    """The `Annotated[list, add]` reducer on `routing_path` must accumulate across
    turns on the same thread — turn 2's final state includes turn 1's nodes."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "t-accum"}}

    r1 = await graph.ainvoke(default_input("Sick leave policy?"), config=cfg)
    turn1_path_len = len(r1["routing_path"])
    assert "llm_answer" in r1["routing_path"]

    r2 = await graph.ainvoke(default_input("Pension contribution?"), config=cfg)
    turn2_path = r2["routing_path"]

    # Reducer applies fresh writes on top of the prior state, so turn 2's path is longer
    # than turn 1's and contains the turn-1 entries as its prefix.
    assert len(turn2_path) > turn1_path_len
    assert turn2_path[:turn1_path_len] == r1["routing_path"]
