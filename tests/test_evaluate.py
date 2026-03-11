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
