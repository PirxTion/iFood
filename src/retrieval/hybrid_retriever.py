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
