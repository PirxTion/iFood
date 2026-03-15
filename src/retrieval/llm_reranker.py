# src/retrieval/llm_reranker.py
import json
import os
import re
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers.cross_encoder import CrossEncoder

from src.config import PROXY_URL, PROXY_KEY, LLM_MODEL, RERANK_TOP_N, FINAL_TOP_K, CROSS_ENCODER_MODEL


class Reranker(ABC):
    """Abstract base class for all rerankers."""

    def __init__(self, items: list[dict]):
        self.item_lookup = {item["item_id"]: item for item in items}

    def _item_text(self, item: dict) -> str:
        """Compact text representation of an item (no ID prefix)."""
        tax = item.get("taxonomy", {})
        tax_str = f"{tax.get('l0', '')}/{tax.get('l1', '')}/{tax.get('l2', '')}"
        return f"{item['name']} | {item['category_name']} | {item['description']} | {tax_str}"

    @abstractmethod
    def rerank(self, query: str, candidate_ids: list[str], top_k: int = FINAL_TOP_K) -> list[str]:
        """Return top_k candidate IDs sorted by relevance to query."""
        ...


def get_client() -> OpenAI:
    key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
    return OpenAI(api_key=key, base_url=PROXY_URL)


def build_rerank_prompt(query: str, items_text: str, top_k: int = FINAL_TOP_K) -> str:
    return f"""You are a search relevance expert for iFood (Brazilian food delivery).

Query: "{query}"

Re-rank these items by relevance to the query. Return the top {top_k} item IDs as a JSON array, most relevant first.

IMPORTANT: If the query contains negation (e.g., "sem peixe" = without fish), items containing the negated ingredient must be excluded entirely.

Items:
{items_text}

Return ONLY a JSON array of item IDs: ["most_relevant_id", "second_id", ...]"""


def parse_rerank_response(response: str) -> list[str]:
    """Extract ordered list of item IDs from LLM response."""
    text = response.strip()

    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    # Fallback
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    return []


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers CrossEncoder (local inference).

    Scores all (query, candidate) pairs in a single batch and returns
    candidates sorted by relevance score.
    """

    def __init__(self, items: list[dict], model: str = CROSS_ENCODER_MODEL):
        super().__init__(items)
        self.model = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        top_k: int = FINAL_TOP_K,
    ) -> list[str]:
        """Re-rank candidates by cross-encoder relevance score."""
        valid_ids = [cid for cid in candidate_ids if cid in self.item_lookup]
        if not valid_ids:
            return []

        pairs = [(query, self._item_text(self.item_lookup[cid])) for cid in valid_ids]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]


class LLMReranker(Reranker):
    def __init__(self, items: list[dict]):
        super().__init__(items)
        self.client = get_client()

    def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        top_k: int = FINAL_TOP_K,
    ) -> list[str]:
        """Re-rank candidate items using LLM."""
        candidates = [self.item_lookup[cid] for cid in candidate_ids if cid in self.item_lookup]
        # LLM needs the ID prefix so it can reference items in its response
        lines = [f"[{item['item_id']}] {self._item_text(item)}" for item in candidates]
        items_text = "\n".join(lines)

        prompt = build_rerank_prompt(query, items_text, top_k)

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
            reranked = parse_rerank_response(resp.choices[0].message.content or "")
            # Filter to only valid IDs and limit
            valid = [rid for rid in reranked if rid in self.item_lookup]
            return valid[:top_k]
        except Exception as e:
            print(f"LLM reranker error: {e}")
            return candidate_ids[:top_k]
