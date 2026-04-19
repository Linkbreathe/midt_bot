from __future__ import annotations
import asyncio
from typing import AsyncIterator

SEED_CORPUS: dict[str, list[dict]] = {
    "feed_policy": [
        {"chunk_id": "pol_parental_1", "text": "Employees are entitled to parental leave under company policy \u00a74.2.", "keywords": ["parental", "leave", "policy"]},
        {"chunk_id": "pol_parental_2", "text": "Parental leave may be taken within 14 months of the child's birth.", "keywords": ["parental", "leave", "birth"]},
        {"chunk_id": "pol_sick_1", "text": "Sick leave policy grants up to 30 paid days per calendar year.", "keywords": ["sick", "leave", "paid"]},
        {"chunk_id": "pol_holidays_1", "text": "Employees accrue 25 days of paid holiday per year.", "keywords": ["holiday", "vacation", "paid"]},
    ],
    "feed_benefits": [
        {"chunk_id": "ben_gym_1", "text": "Company gym membership is partially reimbursed up to DKK 300/month.", "keywords": ["gym", "membership", "reimbursement"]},
        {"chunk_id": "ben_pension_1", "text": "Pension contribution is 12% of gross salary, matched by the employer.", "keywords": ["pension", "retirement", "salary"]},
        {"chunk_id": "ben_health_1", "text": "Supplementary health insurance covers dental and physiotherapy.", "keywords": ["health", "insurance", "dental"]},
    ],
    "feed_collective_agreement": [
        {"chunk_id": "ca_parental_1", "text": "Under the collective agreement, parental leave is extended by 4 weeks beyond statutory.", "keywords": ["parental", "leave", "collective", "agreement"]},
        {"chunk_id": "ca_overtime_1", "text": "Overtime above 37 hours/week is compensated at 150% rate.", "keywords": ["overtime", "compensation", "collective"]},
        {"chunk_id": "ca_notice_1", "text": "Notice period is 3 months for employees over 5 years of tenure.", "keywords": ["notice", "termination", "collective"]},
    ],
}


class MockRetriever:
    """Keyword-score retrieval against SEED_CORPUS."""

    def __init__(self, scope: str):
        self.scope = scope

    def retrieve(self, query: str, top_n: int = 3) -> list[dict]:
        corpus = SEED_CORPUS.get(self.scope, [])
        q_tokens = {t.lower() for t in query.split()}
        scored = []
        for chunk in corpus:
            score = sum(1 for kw in chunk["keywords"] if kw.lower() in q_tokens)
            if score > 0:
                scored.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": score / max(len(chunk["keywords"]), 1),
                    "source": self.scope,
                })
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_n]


class MockLLM:
    """Templated response, streamed token-by-token with a short sleep."""

    def __init__(self, token_delay_ms: int = 20):
        self.token_delay_ms = token_delay_ms

    async def astream(self, prompt: str, sources: list[dict]) -> AsyncIterator[str]:
        if not sources:
            text = "I couldn't find relevant sources in the HR knowledge base for your question."
        else:
            snippets = "; ".join(s["text"] for s in sources[:3])
            text = f"Based on the HR knowledge base: {snippets}"
        for token in text.split(" "):
            await asyncio.sleep(self.token_delay_ms / 1000)
            yield token + " "
