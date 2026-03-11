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
