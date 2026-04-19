import pytest

from hr_assistant.mocks import MockRetriever, MockLLM, SEED_CORPUS


def test_retriever_returns_top_n_by_keyword():
    retriever = MockRetriever(scope="feed_policy")
    chunks = retriever.retrieve("parental leave", top_n=3)
    assert len(chunks) <= 3
    assert all("chunk_id" in c for c in chunks)
    assert all(c["source"] == "feed_policy" for c in chunks)


def test_retriever_missing_scope_returns_empty():
    retriever = MockRetriever(scope="feed_nonexistent")
    assert retriever.retrieve("anything") == []


def test_seed_corpus_has_three_scopes():
    assert {"feed_policy", "feed_benefits", "feed_collective_agreement"}.issubset(
        SEED_CORPUS.keys()
    )


@pytest.mark.asyncio
async def test_llm_streams_tokens():
    llm = MockLLM()
    tokens = []
    async for tok in llm.astream("Tell me about leave", sources=[]):
        tokens.append(tok)
    assert len(tokens) > 3
    joined = "".join(tokens)
    assert len(joined) > 0


@pytest.mark.asyncio
async def test_llm_empty_sources_yields_apology():
    llm = MockLLM()
    chunks: list[str] = []
    async for tok in llm.astream("anything", sources=[]):
        chunks.append(tok)
    reply = "".join(chunks)
    assert "couldn't find" in reply.lower() or "no relevant" in reply.lower()
