import math


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(relevances: list[float], k: int, ideal_relevances: list[float] | None = None) -> float:
    """Normalized DCG at k.

    ideal_relevances: relevance scores of ALL ground-truth items (not just retrieved).
    If provided, the ideal DCG is computed from the top-k of this list, so the metric
    reflects how well the system retrieves the best items from the full corpus, not just
    how well it orders the items it happened to retrieve.
    """
    if not relevances:
        return 0.0
    dcg = _dcg(relevances, k)
    ideal_pool = ideal_relevances if ideal_relevances is not None else relevances
    ideal = _dcg(sorted(ideal_pool, reverse=True), k)
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
    ideal_relevances: list[float] | None = None,
    penalty: float = -3,
) -> float:
    """NDCG at k where violating items get their relevance replaced with penalty.

    ideal_relevances: same as ndcg_at_k — full GT relevances for proper normalization.
    The ideal is computed without applying the penalty (a perfect system has no violations).
    """
    adjusted = [
        penalty if v else r
        for r, v in zip(relevances, violations)
    ]
    return ndcg_at_k(adjusted, k, ideal_relevances=ideal_relevances)


def recall_at_k(relevances: list[float], k: int, total_relevant: int) -> float:
    """Recall at k. Fraction of all relevant items in corpus that appear in top-k."""
    if total_relevant == 0:
        return 0.0
    top_k = relevances[:k]
    return sum(1 for r in top_k if r > 0) / total_relevant


def f1_at_k(relevances: list[float], k: int, total_relevant: int) -> float:
    """F1 at k. Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(relevances, k)
    r = recall_at_k(relevances, k, total_relevant)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
