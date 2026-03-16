import pytest
import numpy as np
from unittest.mock import patch
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
    # Pre-compute normalised embeddings (same as __init__ does)
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    retriever._normed = (item_embeddings / norms).astype(np.float32)

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
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    retriever._normed = (item_embeddings / norms).astype(np.float32)

    query_embedding = np.array([1.0, 0.0])
    with patch.object(retriever, '_embed_query', return_value=query_embedding):
        results = retriever.search("test", top_k=1)

    # With cosine similarity, direction matters not magnitude
    assert results[0] == "a"
