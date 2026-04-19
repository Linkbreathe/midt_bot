import pytest

from hr_assistant.nodes import llm_answer_node, extract_memory_node


@pytest.mark.asyncio
async def test_llm_answer_composes_text_from_sources():
    state = {
        "rewritten_query": "parental leave rules",
        "ranked_chunks": [
            {"chunk_id": "p1", "text": "Parental leave is 24 weeks.", "score": 1.0, "source": "feed_policy"},
        ],
        "system_prompt": "You are HR.",
        "memory_context": {"personal": [], "department": []},
        "config": {"language": "en"},
    }
    out = await llm_answer_node(state)
    assert len(out["response_text"]) > 0
    assert out["sources"][0]["chunk_id"] == "p1"


@pytest.mark.asyncio
async def test_llm_answer_handles_no_sources():
    state = {
        "rewritten_query": "xyzzy",
        "ranked_chunks": [],
        "system_prompt": "",
        "memory_context": {},
        "config": {},
    }
    out = await llm_answer_node(state)
    assert "couldn't find" in out["response_text"].lower()
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_extract_memory_flags_work_statement():
    out = await extract_memory_node({
        "message": "I work in the cardiology department",
        "response_text": "Noted.",
    })
    assert any("cardiology" in c["content"].lower() for c in out["memory_candidates"])
    assert out["memory_candidates"][0]["type"] == "personal"


@pytest.mark.asyncio
async def test_extract_memory_empty_when_no_pattern():
    out = await extract_memory_node({"message": "what time is it", "response_text": "12:00"})
    assert out["memory_candidates"] == []
