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
