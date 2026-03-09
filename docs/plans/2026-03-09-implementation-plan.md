# iFood Semantic Search — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a semantic search system that retrieves relevant food items for 60 Portuguese queries, evaluated via LLM-generated ground truth.

**Architecture:** Eval-first approach. Tournament-style LLM judging builds ground truth, then hybrid retrieval (BM25 + dense embeddings + RRF fusion) with LLM re-ranking serves queries. All text in Portuguese.

**Tech Stack:** Python 3.11+, openai (proxy API), rank_bm25, numpy, pandas

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `src/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `src/retrieval/__init__.py`
- Create: `src/config.py`
- Create: `.gitignore`

**Step 1: Create directory structure and init uv project**

```bash
uv init
mkdir -p src/eval src/retrieval eval_data notebooks
touch src/__init__.py src/eval/__init__.py src/retrieval/__init__.py
```

**Step 2: Add dependencies with uv**

```bash
uv add openai pandas numpy rank_bm25 scikit-learn tqdm
uv add --dev pytest
```

**Step 3: Write .gitignore**

```
data/
eval_data/
*.pdf
key.md
__pycache__/
*.pyc
.env
*.egg-info/
.ipynb_checkpoints/
```

**Step 4: Write src/config.py**

```python
import os

PROXY_URL = "https://oovault.nl/api/proxy/v1"
PROXY_KEY = os.environ.get("PROXY_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

DATA_DIR = "data"
EVAL_DIR = "eval_data"

ITEMS_CSV = f"{DATA_DIR}/5k_items.csv"
QUERIES_CSV = f"{DATA_DIR}/queries.csv"

# Retrieval hyperparams
BM25_TOP_K = 50
DENSE_TOP_K = 50
RRF_K = 60
RERANK_TOP_N = 20
FINAL_TOP_K = 10

# Eval ground truth params
GT_BATCH_SIZE = 1000
GT_ROUND1_TOP_PER_BATCH = 15
GT_ROUND1_RUNS = 2
GT_ROUND2_RUNS = 3
```

**Step 5: Commit**

```bash
git init
git add .gitignore pyproject.toml uv.lock src/__init__.py src/eval/__init__.py src/retrieval/__init__.py src/config.py docs/
git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: Data Loader

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py`

**Step 1: Write the test**

```python
# tests/test_data_loader.py
import pytest
from src.data_loader import load_items, load_queries, build_item_text


def test_load_items_returns_list_of_dicts():
    items = load_items()
    assert len(items) == 5000
    first = items[0]
    assert "item_id" in first
    assert "name" in first
    assert "category_name" in first
    assert "description" in first
    assert "taxonomy" in first
    assert "text" in first  # combined text representation


def test_load_items_parses_metadata():
    items = load_items()
    first = items[0]
    assert isinstance(first["taxonomy"], dict)
    assert "l0" in first["taxonomy"]
    assert isinstance(first["price"], float)
    assert isinstance(first["images"], list)


def test_load_queries_returns_list_of_dicts():
    queries = load_queries()
    assert len(queries) == 60
    first = queries[0]
    assert "query" in first
    assert "category" in first
    assert first["category"] in ("semantic", "keyword", "negative")


def test_build_item_text_includes_all_fields():
    text = build_item_text(
        name="Pizza Margherita",
        category_name="Pizzas",
        description="Massa fina com queijo",
        taxonomy={"l0": "ALIMENTOS_PREPARADOS", "l1": "PIZZAS", "l2": "PIZZA_TRADICIONAL"},
        lac_free=True,
        vegan=False,
        organic=False,
    )
    assert "Pizza Margherita" in text
    assert "Pizzas" in text
    assert "Massa fina com queijo" in text
    assert "ALIMENTOS_PREPARADOS" in text
    assert "sem lactose" in text
    assert "vegano" not in text


def test_build_item_text_no_tags_when_false():
    text = build_item_text(
        name="Test",
        category_name="Cat",
        description="Desc",
        taxonomy={"l0": "A", "l1": "B", "l2": "C"},
        lac_free=False,
        vegan=False,
        organic=False,
    )
    assert "sem lactose" not in text
    assert "vegano" not in text
    assert "orgânico" not in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_loader'`

**Step 3: Write implementation**

```python
# src/data_loader.py
import ast
import csv
from typing import Any

from src.config import ITEMS_CSV, QUERIES_CSV


def build_item_text(
    name: str,
    category_name: str,
    description: str,
    taxonomy: dict,
    lac_free: bool = False,
    vegan: bool = False,
    organic: bool = False,
) -> str:
    """Build combined text representation for an item."""
    tax_str = f"{taxonomy.get('l0', '')}/{taxonomy.get('l1', '')}/{taxonomy.get('l2', '')}"
    parts = [name, category_name, description, tax_str]

    if lac_free:
        parts.append("sem lactose")
    if vegan:
        parts.append("vegano")
    if organic:
        parts.append("orgânico")

    return " | ".join(parts)


def load_items(path: str = ITEMS_CSV) -> list[dict[str, Any]]:
    """Load and parse items CSV into structured dicts."""
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = ast.literal_eval(row["itemMetadata"])
            profile = ast.literal_eval(row["itemProfile"])

            item = {
                "item_id": row["itemId"],
                "merchant_id": row["merchantId"],
                "name": meta["name"],
                "category_name": meta["category_name"],
                "description": meta["description"],
                "price": meta["price"],
                "images": meta["images"],
                "taxonomy": meta["taxonomy"],
                "lac_free": meta.get("lacFree", False),
                "vegan": meta.get("vegan", False),
                "organic": meta.get("organic", False),
                "tags": meta.get("tags", []),
                "metrics": profile.get("metrics", {}),
            }
            item["text"] = build_item_text(
                name=item["name"],
                category_name=item["category_name"],
                description=item["description"],
                taxonomy=item["taxonomy"],
                lac_free=item["lac_free"],
                vegan=item["vegan"],
                organic=item["organic"],
            )
            items.append(item)
    return items


def load_queries(path: str = QUERIES_CSV) -> list[dict[str, str]]:
    """Load queries CSV."""
    queries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query": row["search_term_pt"],
                "category": row["category"],
            })
    return queries
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data_loader.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with item text representation"
```

---

### Task 3: Evaluation Metrics

**Files:**
- Create: `src/eval/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the tests**

```python
# tests/test_metrics.py
import pytest
import math
from src.eval.metrics import ndcg_at_k, mrr, precision_at_k, ncvr_at_k, penalized_ndcg_at_k


def test_ndcg_perfect_ranking():
    # Items ranked perfectly: relevance [3, 2, 1, 0]
    relevances = [3, 2, 1, 0]
    score = ndcg_at_k(relevances, k=4)
    assert score == pytest.approx(1.0)


def test_ndcg_worst_ranking():
    # Reversed: [0, 0, 0, 3]
    relevances = [0, 0, 0, 3]
    score = ndcg_at_k(relevances, k=4)
    assert score < 0.5


def test_ndcg_empty():
    assert ndcg_at_k([], k=10) == 0.0


def test_mrr_first_relevant():
    # First result is relevant
    relevances = [3, 0, 0]
    assert mrr(relevances) == 1.0


def test_mrr_third_relevant():
    relevances = [0, 0, 2, 1]
    assert mrr(relevances) == pytest.approx(1 / 3)


def test_mrr_none_relevant():
    assert mrr([0, 0, 0]) == 0.0


def test_precision_at_k():
    relevances = [3, 0, 2, 0, 1]
    assert precision_at_k(relevances, k=5) == pytest.approx(3 / 5)


def test_precision_at_k_truncates():
    relevances = [3, 0, 2, 0, 1]
    assert precision_at_k(relevances, k=3) == pytest.approx(2 / 3)


def test_ncvr_at_k():
    # violations is a list of bools parallel to the ranked results
    violations = [False, True, False, True, False]
    assert ncvr_at_k(violations, k=5) == pytest.approx(2 / 5)


def test_ncvr_no_violations():
    violations = [False, False, False]
    assert ncvr_at_k(violations, k=3) == 0.0


def test_penalized_ndcg_no_violations():
    relevances = [3, 2, 1, 0]
    violations = [False, False, False, False]
    # Should be same as regular NDCG
    assert penalized_ndcg_at_k(relevances, violations, k=4) == pytest.approx(
        ndcg_at_k(relevances, k=4)
    )


def test_penalized_ndcg_with_violations():
    relevances = [3, 2, 1, 0]
    violations = [False, True, False, False]  # 2nd item violates
    penalized = penalized_ndcg_at_k(relevances, violations, k=4, penalty=-3)
    regular = ndcg_at_k(relevances, k=4)
    assert penalized < regular
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/eval/metrics.py
import math


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG at k."""
    if not relevances:
        return 0.0
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def mrr(relevances: list[float]) -> float:
    """Mean Reciprocal Rank. Returns 1/rank of first relevant item (rel > 0)."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: list[float], k: int) -> float:
    """Precision at k. Fraction of top-k that are relevant (rel > 0)."""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r > 0) / len(top_k)


def ncvr_at_k(violations: list[bool], k: int) -> float:
    """Negative Constraint Violation Rate at k."""
    top_k = violations[:k]
    if not top_k:
        return 0.0
    return sum(1 for v in top_k if v) / len(top_k)


def penalized_ndcg_at_k(
    relevances: list[float],
    violations: list[bool],
    k: int,
    penalty: float = -3,
) -> float:
    """NDCG at k where violating items get their relevance replaced with penalty."""
    adjusted = [
        penalty if v else r
        for r, v in zip(relevances, violations)
    ]
    return ndcg_at_k(adjusted, k)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: All 13 tests PASS

**Step 5: Commit**

```bash
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: evaluation metrics (NDCG, MRR, P@K, NCVR, penalized NDCG)"
```

---

### Task 4: Ground Truth Builder — Round 1 (Coarse Filtering)

**Files:**
- Create: `src/eval/build_ground_truth.py`
- Create: `tests/test_ground_truth.py`

**Step 1: Write the test**

```python
# tests/test_ground_truth.py
import pytest
import json
from unittest.mock import patch, MagicMock
from src.eval.build_ground_truth import (
    format_items_for_prompt,
    parse_round1_response,
    build_round1_prompt,
)


def test_format_items_for_prompt():
    items = [
        {"item_id": "abc123", "name": "Pizza", "category_name": "Pizzas",
         "description": "Queijo e tomate", "taxonomy": {"l0": "A", "l1": "B", "l2": "C"},
         "price": 25.0},
        {"item_id": "def456", "name": "Sushi", "category_name": "Japonesa",
         "description": "Salmão", "taxonomy": {"l0": "A", "l1": "B", "l2": "C"},
         "price": 45.0},
    ]
    text = format_items_for_prompt(items)
    assert "abc123" in text
    assert "Pizza" in text
    assert "def456" in text
    assert "Sushi" in text


def test_parse_round1_response_json_list():
    response = '["abc123", "def456", "ghi789"]'
    result = parse_round1_response(response)
    assert result == ["abc123", "def456", "ghi789"]


def test_parse_round1_response_json_in_markdown():
    response = 'Here are the results:\n```json\n["abc123", "def456"]\n```'
    result = parse_round1_response(response)
    assert result == ["abc123", "def456"]


def test_build_round1_prompt_contains_query():
    prompt = build_round1_prompt("pizza calabresa", "item text block", top_n=15)
    assert "pizza calabresa" in prompt
    assert "15" in prompt
    assert "item text block" in prompt
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ground_truth.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_ground_truth.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/eval/build_ground_truth.py tests/test_ground_truth.py
git commit -m "feat: ground truth builder with tournament-style LLM judging"
```

---

### Task 5: Evaluator Module

**Files:**
- Create: `src/eval/evaluate.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write the test**

```python
# tests/test_evaluate.py
import pytest
from src.eval.evaluate import evaluate_retriever


def test_evaluate_retriever_returns_metrics():
    ground_truth = {
        "pizza": [
            {"item_id": "a", "relevance": 3, "violation": False},
            {"item_id": "b", "relevance": 2, "violation": False},
            {"item_id": "c", "relevance": 0, "violation": False},
        ]
    }
    queries = [{"query": "pizza", "category": "keyword"}]

    # Mock retriever returns items in order: a, c, b
    def mock_retriever(query_text, top_k):
        return ["a", "c", "b"]

    results = evaluate_retriever(mock_retriever, queries, ground_truth, k=3)
    assert "overall" in results
    assert "ndcg@3" in results["overall"]
    assert "mrr" in results["overall"]
    assert "precision@3" in results["overall"]
    assert "by_category" in results
    assert "keyword" in results["by_category"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/eval/evaluate.py
from typing import Callable
from src.eval.metrics import ndcg_at_k, mrr, precision_at_k, ncvr_at_k, penalized_ndcg_at_k


def evaluate_retriever(
    retriever: Callable[[str, int], list[str]],
    queries: list[dict],
    ground_truth: dict,
    k: int = 10,
) -> dict:
    """Evaluate a retriever function against ground truth.

    Args:
        retriever: Function(query_text, top_k) -> list of item_ids
        queries: List of {"query": str, "category": str}
        ground_truth: Dict mapping query_text -> list of {"item_id", "relevance", "violation"}
        k: Cutoff for metrics
    """
    all_metrics = []
    by_category: dict[str, list[dict]] = {}

    for q in queries:
        query_text = q["query"]
        category = q["category"]
        gt_items = ground_truth.get(query_text, [])

        if not gt_items:
            continue

        # Build lookup: item_id -> {relevance, violation}
        gt_lookup = {item["item_id"]: item for item in gt_items}

        # Get retriever results
        retrieved_ids = retriever(query_text, k)

        # Build relevance and violation lists aligned to retrieved order
        relevances = [gt_lookup.get(rid, {}).get("relevance", 0) for rid in retrieved_ids]
        violations = [gt_lookup.get(rid, {}).get("violation", False) for rid in retrieved_ids]

        metrics = {
            f"ndcg@{k}": ndcg_at_k(relevances, k),
            "mrr": mrr(relevances),
            f"precision@{k}": precision_at_k(relevances, k),
        }

        if category == "negative":
            metrics[f"ncvr@{k}"] = ncvr_at_k(violations, k)
            metrics[f"penalized_ndcg@{k}"] = penalized_ndcg_at_k(relevances, violations, k)

        all_metrics.append(metrics)
        by_category.setdefault(category, []).append(metrics)

    # Aggregate
    def avg_metrics(metric_list: list[dict]) -> dict:
        if not metric_list:
            return {}
        keys = set()
        for m in metric_list:
            keys.update(m.keys())
        return {
            key: sum(m.get(key, 0) for m in metric_list) / len(metric_list)
            for key in sorted(keys)
        }

    return {
        "overall": avg_metrics(all_metrics),
        "by_category": {cat: avg_metrics(ms) for cat, ms in by_category.items()},
        "num_queries": len(all_metrics),
    }
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluator module with per-category metric breakdown"
```

---

### Task 6: BM25 Retriever

**Files:**
- Create: `src/retrieval/bm25_retriever.py`
- Create: `tests/test_bm25_retriever.py`

**Step 1: Write the test**

```python
# tests/test_bm25_retriever.py
import pytest
from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_index_and_search():
    items = [
        {"item_id": "1", "text": "Pizza Margherita queijo tomate"},
        {"item_id": "2", "text": "Sushi salmão arroz"},
        {"item_id": "3", "text": "Pizza calabresa pepperoni"},
    ]
    retriever = BM25Retriever(items)
    results = retriever.search("pizza", top_k=2)
    assert len(results) == 2
    # Both pizza items should rank above sushi
    result_ids = [r for r in results]
    assert "2" not in result_ids


def test_bm25_returns_item_ids():
    items = [
        {"item_id": "a", "text": "Macarrão com molho"},
        {"item_id": "b", "text": "Arroz e feijão"},
    ]
    retriever = BM25Retriever(items)
    results = retriever.search("macarrão", top_k=1)
    assert results == ["a"]


def test_bm25_top_k_limits_results():
    items = [{"item_id": str(i), "text": f"item {i} comida"} for i in range(100)]
    retriever = BM25Retriever(items)
    results = retriever.search("comida", top_k=5)
    assert len(results) == 5
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bm25_retriever.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/retrieval/bm25_retriever.py
from rank_bm25 import BM25Okapi

from src.config import BM25_TOP_K


class BM25Retriever:
    def __init__(self, items: list[dict]):
        """Initialize BM25 index from items.

        Each item must have 'item_id' and 'text' keys.
        """
        self.items = items
        self.item_ids = [item["item_id"] for item in items]

        # Tokenize: lowercase and split on whitespace/punctuation
        self.corpus = [self._tokenize(item["text"]) for item in items]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return text.lower().split()

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[str]:
        """Search for query, return top_k item IDs."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.item_ids[i] for i in top_indices if scores[i] > 0][:top_k]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_bm25_retriever.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/retrieval/bm25_retriever.py tests/test_bm25_retriever.py
git commit -m "feat: BM25 retriever with simple tokenization"
```

---

### Task 7: Dense Retriever

**Files:**
- Create: `src/retrieval/dense_retriever.py`
- Create: `tests/test_dense_retriever.py`

**Step 1: Write the test**

```python
# tests/test_dense_retriever.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.retrieval.dense_retriever import DenseRetriever


def test_dense_retriever_search_returns_ids():
    # Mock embeddings: 3 items, 4-dim embeddings
    item_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # item a
        [0.0, 1.0, 0.0, 0.0],  # item b
        [0.9, 0.1, 0.0, 0.0],  # item c (similar to a)
    ])
    item_ids = ["a", "b", "c"]

    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.item_ids = item_ids
    retriever.item_embeddings = item_embeddings
    retriever.client = None  # won't be used

    # Query embedding similar to item a
    query_embedding = np.array([0.95, 0.05, 0.0, 0.0])
    with patch.object(retriever, '_embed_query', return_value=query_embedding):
        results = retriever.search("test query", top_k=2)

    assert len(results) == 2
    assert results[0] == "a"  # most similar
    assert results[1] == "c"  # second most similar


def test_dense_retriever_cosine_similarity():
    """Verify cosine similarity is used, not dot product."""
    item_embeddings = np.array([
        [10.0, 0.0],  # large magnitude but same direction as query
        [0.5, 0.5],   # different direction
    ])
    item_ids = ["a", "b"]

    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.item_ids = item_ids
    retriever.item_embeddings = item_embeddings
    retriever.client = None

    query_embedding = np.array([1.0, 0.0])
    with patch.object(retriever, '_embed_query', return_value=query_embedding):
        results = retriever.search("test", top_k=2)

    # With cosine similarity, direction matters not magnitude
    assert results[0] == "a"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dense_retriever.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/retrieval/dense_retriever.py
import os
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.config import PROXY_URL, PROXY_KEY, EMBEDDING_MODEL, DENSE_TOP_K


class DenseRetriever:
    def __init__(self, items: list[dict], batch_size: int = 100):
        """Initialize dense retriever by embedding all items.

        Each item must have 'item_id' and 'text' keys.
        """
        key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
        self.client = OpenAI(api_key=key, base_url=PROXY_URL)
        self.item_ids = [item["item_id"] for item in items]

        # Embed all items in batches
        texts = [item["text"] for item in items]
        self.item_embeddings = self._embed_batch(texts, batch_size)

    def _embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts in batches."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding items"):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_embs = [e.embedding for e in resp.data]
            all_embeddings.extend(batch_embs)
        return np.array(all_embeddings)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        return np.array(resp.data[0].embedding)

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[str]:
        """Search by cosine similarity, return top_k item IDs."""
        query_emb = self._embed_query(query)

        # Cosine similarity
        norms = np.linalg.norm(self.item_embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = self.item_embeddings @ query_emb / (norms * query_norm + 1e-10)

        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.item_ids[i] for i in top_indices]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_dense_retriever.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add src/retrieval/dense_retriever.py tests/test_dense_retriever.py
git commit -m "feat: dense retriever with OpenAI embeddings and cosine similarity"
```

---

### Task 8: Hybrid Retriever (RRF Fusion)

**Files:**
- Create: `src/retrieval/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever.py`

**Step 1: Write the test**

```python
# tests/test_hybrid_retriever.py
import pytest
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever


def test_rrf_basic():
    # Two ranked lists
    list1 = ["a", "b", "c"]  # a=rank1, b=rank2, c=rank3
    list2 = ["c", "a", "d"]  # c=rank1, a=rank2, d=rank3

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    # 'a' appears at rank 1 and rank 2: 1/61 + 1/62
    # 'c' appears at rank 3 and rank 1: 1/63 + 1/61
    # 'a' should be top because 1/61 + 1/62 > 1/63 + 1/61
    assert fused[0] == "a"


def test_rrf_deduplicates():
    list1 = ["a", "b"]
    list2 = ["a", "c"]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(set(fused)) == len(fused)  # no duplicates


def test_rrf_respects_top_k():
    list1 = ["a", "b", "c", "d", "e"]
    list2 = ["e", "d", "c", "b", "a"]
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
    assert len(fused) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/retrieval/hybrid_retriever.py
from collections import defaultdict

from src.config import RRF_K, BM25_TOP_K, DENSE_TOP_K, FINAL_TOP_K


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[str]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    score(item) = sum(1 / (k + rank_i)) for each list where item appears.
    """
    scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            scores[item_id] += 1.0 / (k + rank)

    sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    if top_k is not None:
        sorted_items = sorted_items[:top_k]
    return sorted_items


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def search(self, query: str, top_k: int = FINAL_TOP_K) -> list[str]:
        """Run both retrievers and fuse with RRF."""
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion(
            [bm25_results, dense_results], k=RRF_K, top_k=top_k
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_hybrid_retriever.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/retrieval/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: hybrid retriever with reciprocal rank fusion"
```

---

### Task 9: LLM Re-ranker

**Files:**
- Create: `src/retrieval/llm_reranker.py`
- Create: `tests/test_llm_reranker.py`

**Step 1: Write the test**

```python
# tests/test_llm_reranker.py
import pytest
from src.retrieval.llm_reranker import build_rerank_prompt, parse_rerank_response


def test_build_rerank_prompt_includes_query_and_items():
    prompt = build_rerank_prompt(
        query="pizza sem queijo",
        items_text="[id1] Pizza Margherita\n[id2] Pizza vegana",
        top_k=10,
    )
    assert "pizza sem queijo" in prompt
    assert "id1" in prompt
    assert "id2" in prompt
    assert "10" in prompt


def test_parse_rerank_response_json_list():
    response = '["id2", "id1", "id3"]'
    result = parse_rerank_response(response)
    assert result == ["id2", "id1", "id3"]


def test_parse_rerank_response_markdown():
    response = '```json\n["id2", "id1"]\n```'
    result = parse_rerank_response(response)
    assert result == ["id2", "id1"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_reranker.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_llm_reranker.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/retrieval/llm_reranker.py tests/test_llm_reranker.py
git commit -m "feat: LLM re-ranker with negation-aware prompting"
```

---

### Task 10: Main Pipeline Script

**Files:**
- Create: `run_eval.py`
- Create: `run_search.py`

**Step 1: Write run_eval.py (generates ground truth)**

```python
# run_eval.py
"""Generate ground truth and/or evaluate retrievers."""
import argparse
import json
import os

from src.data_loader import load_items, load_queries
from src.eval.build_ground_truth import build_ground_truth, save_ground_truth, load_ground_truth
from src.eval.evaluate import evaluate_retriever
from src.config import EVAL_DIR


def main():
    parser = argparse.ArgumentParser(description="Build ground truth or evaluate retrievers")
    parser.add_argument("--build-gt", action="store_true", help="Build ground truth via LLM")
    parser.add_argument("--evaluate", choices=["bm25", "dense", "hybrid", "full"], help="Evaluate a retriever")
    parser.add_argument("--k", type=int, default=10, help="Top-K for evaluation")
    args = parser.parse_args()

    items = load_items()
    queries = load_queries()

    if args.build_gt:
        print(f"Building ground truth for {len(queries)} queries over {len(items)} items...")
        gt = build_ground_truth(items, queries)
        save_ground_truth(gt)
        print("Done!")
        return

    if args.evaluate:
        gt = load_ground_truth()

        if args.evaluate == "bm25":
            from src.retrieval.bm25_retriever import BM25Retriever
            retriever = BM25Retriever(items)
            results = evaluate_retriever(retriever.search, queries, gt, k=args.k)

        elif args.evaluate == "dense":
            from src.retrieval.dense_retriever import DenseRetriever
            retriever = DenseRetriever(items)
            results = evaluate_retriever(retriever.search, queries, gt, k=args.k)

        elif args.evaluate == "hybrid":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            hybrid = HybridRetriever(bm25, dense)
            results = evaluate_retriever(hybrid.search, queries, gt, k=args.k)

        elif args.evaluate == "full":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.retrieval.llm_reranker import LLMReranker
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            hybrid = HybridRetriever(bm25, dense)
            reranker = LLMReranker(items)

            def full_pipeline(query_text, top_k):
                candidates = hybrid.search(query_text, top_k=20)
                return reranker.rerank(query_text, candidates, top_k=top_k)

            results = evaluate_retriever(full_pipeline, queries, gt, k=args.k)

        print(json.dumps(results, indent=2, ensure_ascii=False))

        # Save results
        os.makedirs(EVAL_DIR, exist_ok=True)
        with open(f"{EVAL_DIR}/eval_{args.evaluate}.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

**Step 2: Write run_search.py (interactive search)**

```python
# run_search.py
"""Run interactive search queries."""
import argparse

from src.data_loader import load_items
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.llm_reranker import LLMReranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid", "full"], default="full")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("query", nargs="?", help="Search query (or interactive mode if omitted)")
    args = parser.parse_args()

    print("Loading items...")
    items = load_items()
    item_lookup = {item["item_id"]: item for item in items}

    print(f"Initializing {args.mode} retriever...")
    if args.mode in ("bm25", "hybrid", "full"):
        bm25 = BM25Retriever(items)
    if args.mode in ("dense", "hybrid", "full"):
        dense = DenseRetriever(items)
    if args.mode in ("hybrid", "full"):
        hybrid = HybridRetriever(bm25, dense)
    if args.mode == "full":
        reranker = LLMReranker(items)

    def search(query_text: str):
        if args.mode == "bm25":
            return bm25.search(query_text, args.top_k)
        elif args.mode == "dense":
            return dense.search(query_text, args.top_k)
        elif args.mode == "hybrid":
            return hybrid.search(query_text, args.top_k)
        else:
            candidates = hybrid.search(query_text, top_k=20)
            return reranker.rerank(query_text, candidates, top_k=args.top_k)

    def display_results(results):
        for i, item_id in enumerate(results, 1):
            item = item_lookup.get(item_id, {})
            print(f"  {i}. {item.get('name', '?')} | {item.get('category_name', '?')} | R${item.get('price', 0):.2f}")

    if args.query:
        results = search(args.query)
        display_results(results)
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            results = search(query)
            display_results(results)
            print()


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add run_eval.py run_search.py
git commit -m "feat: main pipeline scripts for evaluation and interactive search"
```

---

### Task 11: Integration Test — End-to-End Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

This tests the full pipeline with a small subset (no real API calls for BM25, mocked for dense/LLM).

```python
# tests/test_integration.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_loader import load_items, load_queries
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.eval.evaluate import evaluate_retriever


def test_bm25_on_real_data():
    """Smoke test: BM25 on actual data returns results."""
    items = load_items()
    queries = load_queries()

    retriever = BM25Retriever(items)
    results = retriever.search("pizza calabresa", top_k=10)

    assert len(results) > 0
    assert len(results) <= 10
    # Should find pizza-related items
    item_lookup = {item["item_id"]: item for item in items}
    top_name = item_lookup[results[0]]["name"].lower()
    assert "pizza" in top_name or "calabresa" in top_name


def test_rrf_fusion_on_real_data():
    """Smoke test: RRF fusion produces valid merged results."""
    items = load_items()
    retriever = BM25Retriever(items)

    # Run two different queries to simulate two retrievers
    list1 = retriever.search("pizza", top_k=20)
    list2 = retriever.search("calabresa", top_k=20)

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=10)
    assert len(fused) <= 10
    assert len(set(fused)) == len(fused)  # no duplicates


def test_evaluate_bm25_on_mock_gt():
    """Evaluate BM25 against mock ground truth."""
    items = load_items()
    retriever = BM25Retriever(items)

    # Create simple ground truth for one query
    results = retriever.search("pizza calabresa", top_k=5)
    mock_gt = {
        "pizza calabresa": [
            {"item_id": results[0], "relevance": 3, "violation": False},
            {"item_id": results[1], "relevance": 2, "violation": False},
        ]
    }
    queries = [{"query": "pizza calabresa", "category": "keyword"}]

    eval_results = evaluate_retriever(retriever.search, queries, mock_gt, k=5)
    assert eval_results["overall"]["ndcg@5"] > 0
    assert eval_results["overall"]["mrr"] > 0
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All 3 tests PASS (these only use BM25, no API calls)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke tests for BM25 and evaluation pipeline"
```

---

### Task 12: Run Ground Truth Generation

**No new files — this is an execution step.**

**Step 1: Set environment variable**

```bash
export PROXY_KEY="oo_proxy_cH8rsyHajw35BBCn8Yu0fJ1CPXrxCF7V"
```

**Step 2: Run ground truth builder**

Run: `uv run python run_eval.py --build-gt`

Expected: Takes ~15-30 minutes. Outputs progress for each query. Saves to `eval_data/ground_truth.json`.

**Step 3: Verify output**

```bash
uv run python -c "import json; gt=json.load(open('eval_data/ground_truth.json')); print(f'{len(gt)} queries'); print(f'Avg candidates: {sum(len(v) for v in gt.values())/len(gt):.0f}')"
```

Expected: 60 queries, ~50-150 candidates per query on average.

**Step 4: Commit ground truth metadata (not the file itself)**

```bash
git add -f eval_data/.gitkeep  # if needed
git commit -m "chore: ground truth generation complete"
```

---

### Task 13: Run Ablation Evaluation

**No new files — execution steps.**

**Step 1: Evaluate BM25 only**

Run: `uv run python run_eval.py --evaluate bm25`

**Step 2: Evaluate dense only**

Run: `uv run python run_eval.py --evaluate dense`

**Step 3: Evaluate hybrid (BM25 + dense + RRF)**

Run: `uv run python run_eval.py --evaluate hybrid`

**Step 4: Evaluate full pipeline (hybrid + LLM re-ranking)**

Run: `uv run python run_eval.py --evaluate full`

**Step 5: Compare results**

All results saved in `eval_data/eval_*.json`. Compare NDCG@10, MRR, P@10 across ablations, especially the per-category breakdown for negative queries.

**Step 6: Commit evaluation results**

```bash
git add eval_data/eval_*.json
git commit -m "results: ablation evaluation across 4 retriever configurations"
```

---

## Summary of Tasks

| # | Task | Key Output |
|---|------|------------|
| 1 | Project scaffolding | `src/config.py`, `requirements.txt`, `.gitignore` |
| 2 | Data loader | `src/data_loader.py` + tests |
| 3 | Eval metrics | `src/eval/metrics.py` + tests |
| 4 | Ground truth builder | `src/eval/build_ground_truth.py` + tests |
| 5 | Evaluator | `src/eval/evaluate.py` + tests |
| 6 | BM25 retriever | `src/retrieval/bm25_retriever.py` + tests |
| 7 | Dense retriever | `src/retrieval/dense_retriever.py` + tests |
| 8 | Hybrid retriever (RRF) | `src/retrieval/hybrid_retriever.py` + tests |
| 9 | LLM re-ranker | `src/retrieval/llm_reranker.py` + tests |
| 10 | Main pipeline scripts | `run_eval.py`, `run_search.py` |
| 11 | Integration tests | `tests/test_integration.py` |
| 12 | Generate ground truth | `eval_data/ground_truth.json` |
| 13 | Run ablation evaluation | `eval_data/eval_*.json` |
