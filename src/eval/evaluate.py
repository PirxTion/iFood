from typing import Callable
from src.eval.metrics import ndcg_at_k, mrr, precision_at_k, ncvr_at_k, penalized_ndcg_at_k, recall_at_k, f1_at_k


RELEVANCE_THRESHOLD = 1  # Items with relevance <= this are treated as non-relevant for binary metrics


def evaluate_retriever(
    retriever: Callable[[str, int], list[str]],
    queries: list[dict],
    ground_truth: dict,
    k: int = 10,
    collector: "QueryTraceCollector | None" = None,
) -> dict:
    """Evaluate a retriever function against ground truth.

    Args:
        retriever: Function(query_text, top_k) -> list of item_ids
        queries: List of {"query": str, "category": str}
        ground_truth: Dict mapping query_text -> list of {"item_id", "relevance", "violation"}
        k: Cutoff for metrics

    Relevance threshold: only items with relevance > RELEVANCE_THRESHOLD count as relevant
    for binary metrics (precision, recall, f1). nDCG uses the full graded relevance.
    """
    all_metrics = []
    by_category: dict[str, list[dict]] = {}

    for q in queries:
        query_text = q["query"]
        category = q["category"]
        gt_items = ground_truth.get(query_text, [])

        # Count relevant items above threshold (used as recall denominator)
        total_relevant = sum(1 for item in gt_items if item.get("relevance", 0) > RELEVANCE_THRESHOLD)
        if total_relevant == 0:
            continue

        # Build lookup: item_id -> {relevance, violation}
        gt_lookup = {item["item_id"]: item for item in gt_items}

        # Full GT relevances for ideal DCG — normalises against the best possible retrieval
        gt_relevances = [item.get("relevance", 0) for item in gt_items]

        # Get retriever results
        retrieved_ids = retriever(query_text, k)

        # Build relevance and violation lists aligned to retrieved order
        relevances = [gt_lookup.get(rid, {}).get("relevance", 0) for rid in retrieved_ids]
        violations = [gt_lookup.get(rid, {}).get("violation", False) for rid in retrieved_ids]
        # Binary relevance for precision/recall/f1 — apply the threshold
        binary_relevances = [r if r > RELEVANCE_THRESHOLD else 0 for r in relevances]

        metrics = {
            f"ndcg@{k}": ndcg_at_k(relevances, k, ideal_relevances=gt_relevances),
            "mrr": mrr(binary_relevances),
            f"precision@{k}": precision_at_k(binary_relevances, k),
            f"recall@{k}": recall_at_k(binary_relevances, k, total_relevant),
            f"f1@{k}": f1_at_k(binary_relevances, k, total_relevant),
        }

        if category == "negative":
            metrics[f"ncvr@{k}"] = ncvr_at_k(violations, k)
            metrics[f"penalized_ndcg@{k}"] = penalized_ndcg_at_k(binary_relevances, violations, k, ideal_relevances=gt_relevances)

        if collector is not None:
            collector.set_query_metrics(query_text, metrics)

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
