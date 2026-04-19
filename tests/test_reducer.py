from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import pytest

from hr_assistant.state import HRState


@pytest.mark.asyncio
async def test_state_reducer_merges_parallel_chunks():
    """Two parallel nodes writing to retrieved_chunks should have their outputs concatenated."""

    async def emitter_a(state: HRState) -> dict:
        return {"retrieved_chunks": [{"source": "a", "chunk_id": "a1", "text": "alpha", "score": 0.9}]}

    async def emitter_b(state: HRState) -> dict:
        return {"retrieved_chunks": [{"source": "b", "chunk_id": "b1", "text": "beta", "score": 0.8}]}

    async def fanout(state: HRState):
        return [Send("a", state), Send("b", state)]

    builder = StateGraph(HRState)
    builder.add_node("a", emitter_a)
    builder.add_node("b", emitter_b)
    builder.add_conditional_edges(START, fanout, ["a", "b"])
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    graph = builder.compile()

    result = await graph.ainvoke({"message": "hi", "retrieved_chunks": []})

    assert len(result["retrieved_chunks"]) == 2
    ids = {c["chunk_id"] for c in result["retrieved_chunks"]}
    assert ids == {"a1", "b1"}
