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
