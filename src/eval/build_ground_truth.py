# src/eval/build_ground_truth.py
import json
import os
import re
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.config import (
    PROXY_URL,
    PROXY_KEY,
    LLM_MODEL,
    EVAL_DIR,
    GT_BATCH_SIZE,
    GT_ROUND1_TOP_PER_BATCH,
    GT_ROUND1_RUNS,
    GT_ROUND2_RUNS,
)


def get_client() -> OpenAI:
    key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
    return OpenAI(api_key=key, base_url=PROXY_URL)


def format_items_for_prompt(items: list[dict]) -> str:
    """Format a list of items into a compact text block for the LLM."""
    lines = []
    for item in items:
        tax = item.get("taxonomy", {})
        tax_str = f"{tax.get('l0', '')}/{tax.get('l1', '')}/{tax.get('l2', '')}"
        line = f"[{item['item_id']}] {item['name']} | {item['category_name']} | {item['description']} | {tax_str} | R${item.get('price', 0):.2f}"
        lines.append(line)
    return "\n".join(lines)


def build_round1_prompt(query: str, items_text: str, top_n: int = 15) -> str:
    return f"""You are evaluating search relevance for a food delivery app (iFood, Brazil).

Query: "{query}"

Below are items from the catalog. Select the {top_n} most relevant items for this query.
Return ONLY a JSON array of item IDs, like: ["id1", "id2", ...]

Items:
{items_text}"""


def parse_round1_response(response: str) -> list[str]:
    """Extract list of item IDs from LLM response."""
    # Try direct JSON parse
    text = response.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    # Fallback: find anything that looks like a JSON array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    return []


def build_round2_prompt(query: str, query_category: str, items_text: str) -> str:
    negation_instruction = ""
    if query_category == "negative":
        negation_instruction = """
IMPORTANT: This is a NEGATION query. The user explicitly does NOT want certain ingredients/attributes.
For each item, check if it violates the negation constraint. Mark violation=true if the item contains
what the user explicitly excluded."""

    return f"""You are evaluating search relevance for a food delivery app (iFood, Brazil).

Query: "{query}"
Query type: {query_category}
{negation_instruction}

Score each item's relevance to the query:
- 3 = highly relevant (exactly what the user wants)
- 2 = relevant (good match, minor differences)
- 1 = marginally relevant (loosely related)
- 0 = irrelevant

Return a JSON array of objects: [{{"id": "item_id", "score": 0-3, "violation": true/false}}]
Set violation=true ONLY if the item contradicts an explicit negation in the query.

Items:
{items_text}"""


def parse_round2_response(response: str) -> list[dict]:
    """Extract scored items from LLM response."""
    text = response.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: find JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def run_round1(client: OpenAI, query: str, items: list[dict]) -> set[str]:
    """Round 1: coarse filtering. Returns set of candidate item IDs."""
    candidates = set()

    # Split items into batches
    batches = [items[i:i + GT_BATCH_SIZE] for i in range(0, len(items), GT_BATCH_SIZE)]

    for batch in batches:
        items_text = format_items_for_prompt(batch)
        prompt = build_round1_prompt(query, items_text, top_n=GT_ROUND1_TOP_PER_BATCH)

        for run in range(GT_ROUND1_RUNS):
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                ids = parse_round1_response(resp.choices[0].message.content or "")
                candidates.update(ids)
            except Exception as e:
                print(f"  Round 1 error (run {run}): {e}")
                time.sleep(2)

    return candidates


def run_round2(
    client: OpenAI,
    query: str,
    query_category: str,
    candidate_items: list[dict],
) -> list[dict]:
    """Round 2: fine-grained scoring. Returns list of {id, score, violation}."""
    items_text = format_items_for_prompt(candidate_items)
    prompt = build_round2_prompt(query, query_category, items_text)

    all_scores: dict[str, list[int]] = {}
    all_violations: dict[str, list[bool]] = {}

    for run in range(GT_ROUND2_RUNS):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            scored = parse_round2_response(resp.choices[0].message.content or "")
            for entry in scored:
                item_id = str(entry.get("id", ""))
                score = int(entry.get("score", 0))
                violation = bool(entry.get("violation", False))
                all_scores.setdefault(item_id, []).append(score)
                all_violations.setdefault(item_id, []).append(violation)
        except Exception as e:
            print(f"  Round 2 error (run {run}): {e}")
            time.sleep(2)

    # Compute median score and majority-vote violation
    results = []
    for item_id in all_scores:
        scores = sorted(all_scores[item_id])
        median_score = scores[len(scores) // 2]
        violation = sum(all_violations.get(item_id, [])) > len(all_violations.get(item_id, [])) / 2
        results.append({
            "item_id": item_id,
            "relevance": median_score,
            "violation": violation,
        })

    # Sort by relevance descending
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results


def build_ground_truth(items: list[dict], queries: list[dict]) -> dict:
    """Build complete ground truth for all queries."""
    client = get_client()
    item_lookup = {item["item_id"]: item for item in items}
    ground_truth = {}

    for q in tqdm(queries, desc="Building ground truth"):
        query_text = q["query"]
        query_cat = q["category"]
        print(f"\nProcessing: '{query_text}' ({query_cat})")

        # Round 1: coarse filtering
        candidate_ids = run_round1(client, query_text, items)
        print(f"  Round 1: {len(candidate_ids)} candidates")

        # Get full item dicts for candidates
        candidate_items = [item_lookup[cid] for cid in candidate_ids if cid in item_lookup]

        if not candidate_items:
            print(f"  WARNING: No valid candidates found for '{query_text}'")
            ground_truth[query_text] = []
            continue

        # Round 2: fine-grained scoring
        scored = run_round2(client, query_text, query_cat, candidate_items)
        print(f"  Round 2: {len(scored)} items scored")

        ground_truth[query_text] = scored

    return ground_truth


def save_ground_truth(ground_truth: dict, path: str | None = None):
    """Save ground truth to JSON."""
    if path is None:
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, "ground_truth.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"Ground truth saved to {path}")


def load_ground_truth(path: str | None = None) -> dict:
    """Load ground truth from JSON."""
    if path is None:
        path = os.path.join(EVAL_DIR, "ground_truth.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
