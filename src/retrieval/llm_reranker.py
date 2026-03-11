# src/retrieval/llm_reranker.py
import json
import os
import re

from openai import OpenAI

from src.config import PROXY_URL, PROXY_KEY, LLM_MODEL, RERANK_TOP_N, FINAL_TOP_K


def get_client() -> OpenAI:
    key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
    return OpenAI(api_key=key, base_url=PROXY_URL)


def build_rerank_prompt(query: str, items_text: str, top_k: int = FINAL_TOP_K) -> str:
    return f"""You are a search relevance expert for iFood (Brazilian food delivery).

Query: "{query}"

Re-rank these items by relevance to the query. Return the top {top_k} item IDs as a JSON array, most relevant first.

IMPORTANT: If the query contains negation (e.g., "sem peixe" = without fish), items containing the negated ingredient must be ranked LAST or excluded entirely.

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


class LLMReranker:
    def __init__(self, items: list[dict]):
        """Initialize with full item list for lookup."""
        self.client = get_client()
        self.item_lookup = {item["item_id"]: item for item in items}

    def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        top_k: int = FINAL_TOP_K,
    ) -> list[str]:
        """Re-rank candidate items using LLM."""
        # Build items text for candidates
        candidates = [self.item_lookup[cid] for cid in candidate_ids if cid in self.item_lookup]
        lines = []
        for item in candidates:
            tax = item.get("taxonomy", {})
            tax_str = f"{tax.get('l0', '')}/{tax.get('l1', '')}/{tax.get('l2', '')}"
            line = f"[{item['item_id']}] {item['name']} | {item['category_name']} | {item['description']} | {tax_str}"
            lines.append(line)
        items_text = "\n".join(lines)

        prompt = build_rerank_prompt(query, items_text, top_k)

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            reranked = parse_rerank_response(resp.choices[0].message.content or "")
            # Filter to only valid IDs and limit
            valid = [rid for rid in reranked if rid in self.item_lookup]
            return valid[:top_k]
        except Exception as e:
            print(f"LLM reranker error: {e}")
            return candidate_ids[:top_k]
