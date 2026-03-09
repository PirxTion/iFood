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
