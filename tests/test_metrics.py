import pytest
import math
from src.eval.metrics import (
    ndcg_at_k, mrr, precision_at_k, ncvr_at_k, penalized_ndcg_at_k,
    recall_at_k, f1_at_k,
)


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


# --- recall_at_k tests ---

def test_recall_at_k_all_retrieved():
    # Retrieved 3 relevant out of 3 total relevant
    relevances = [3, 2, 1, 0, 0]
    assert recall_at_k(relevances, k=5, total_relevant=3) == pytest.approx(1.0)


def test_recall_at_k_partial():
    # Retrieved 2 relevant in top-3, but 5 relevant exist in corpus
    relevances = [3, 0, 2, 0, 1]
    assert recall_at_k(relevances, k=3, total_relevant=5) == pytest.approx(2 / 5)


def test_recall_at_k_none_relevant():
    relevances = [0, 0, 0]
    assert recall_at_k(relevances, k=3, total_relevant=4) == 0.0


def test_recall_at_k_zero_total():
    # No relevant items in corpus — edge case
    relevances = [0, 0]
    assert recall_at_k(relevances, k=2, total_relevant=0) == 0.0


# --- f1_at_k tests ---

def test_f1_at_k_perfect():
    # top-3 has 3 relevant, and there are exactly 3 relevant in corpus
    # P@3 = 1.0, R@3 = 1.0, F1 = 1.0
    relevances = [3, 2, 1]
    assert f1_at_k(relevances, k=3, total_relevant=3) == pytest.approx(1.0)


def test_f1_at_k_balanced():
    # top-4: [rel, irr, rel, irr] → P@4 = 0.5
    # total_relevant=4 → R@4 = 2/4 = 0.5
    # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    relevances = [1, 0, 1, 0]
    assert f1_at_k(relevances, k=4, total_relevant=4) == pytest.approx(0.5)


def test_f1_at_k_zero_precision_and_recall():
    relevances = [0, 0, 0]
    assert f1_at_k(relevances, k=3, total_relevant=5) == 0.0


def test_f1_at_k_zero_total_relevant():
    relevances = [0, 0]
    assert f1_at_k(relevances, k=2, total_relevant=0) == 0.0
