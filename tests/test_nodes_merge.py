import pytest

from hr_assistant.nodes import merge_and_rank_node


@pytest.mark.asyncio
async def test_merge_dedupes_by_chunk_id():
    state = {"retrieved_chunks": [
        {"chunk_id": "a", "text": "A", "score": 0.3, "source": "x"},
        {"chunk_id": "a", "text": "A", "score": 0.3, "source": "x"},
        {"chunk_id": "b", "text": "B", "score": 0.9, "source": "y"},
    ]}
    out = await merge_and_rank_node(state)
    assert len(out["ranked_chunks"]) == 2
    assert out["ranked_chunks"][0]["chunk_id"] == "b"  # highest score first


@pytest.mark.asyncio
async def test_merge_caps_at_five():
    state = {"retrieved_chunks": [
        {"chunk_id": f"c{i}", "text": "t", "score": i / 10, "source": "s"}
        for i in range(10)
    ]}
    out = await merge_and_rank_node(state)
    assert len(out["ranked_chunks"]) == 5


@pytest.mark.asyncio
async def test_merge_tolerates_empty():
    out = await merge_and_rank_node({"retrieved_chunks": []})
    assert out["ranked_chunks"] == []
