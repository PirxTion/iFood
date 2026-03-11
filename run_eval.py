"""Generate ground truth and/or evaluate retrievers."""
import argparse
import json
import os
import time

from src.data_loader import load_items, load_queries
from src.eval.build_ground_truth import build_ground_truth, save_ground_truth, load_ground_truth
from src.eval.evaluate import evaluate_retriever
from src.eval.tracing import QueryTraceCollector
from src.eval.run_report import build_run_report, save_run_report
from src.config import (
    EVAL_DIR, BM25_TOP_K, DENSE_TOP_K, RRF_K, RERANK_TOP_N,
    FINAL_TOP_K, EMBEDDING_MODEL, LLM_MODEL,
)


def _query_category(queries, query_text):
    """Look up the category for a query text."""
    return next((q["category"] for q in queries if q["query"] == query_text), "unknown")


def main():
    parser = argparse.ArgumentParser(description="Build ground truth or evaluate retrievers")
    parser.add_argument("--build-gt", action="store_true", help="Build ground truth via LLM")
    parser.add_argument(
        "--evaluate",
        choices=["bm25", "dense", "hybrid", "full", "hf"],
        help="Evaluate a retriever (full=hybrid+LLM reranker, hf=hybrid+HF cross-encoder reranker)",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-K for evaluation")
    args = parser.parse_args()

    items = load_items()
    queries = load_queries()

    if args.build_gt:
        print(f"Building ground truth for {len(queries)} queries over {len(items)} items...")
        gt = build_ground_truth(items, queries)
        save_ground_truth(gt)
        print("Done!")
        return

    if args.evaluate:
        gt = load_ground_truth()
        collector = QueryTraceCollector()

        if args.evaluate == "bm25":
            from src.retrieval.bm25_retriever import BM25Retriever
            retriever = BM25Retriever(items)

            def bm25_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                with collector.stage("bm25") as st:
                    results = retriever.search(query_text, top_k)
                    st.output_ids = results
                collector.finalize_query(results, (time.perf_counter() - t0) * 1000)
                return results

            results = evaluate_retriever(bm25_pipeline, queries, gt, k=args.k, collector=collector)

        elif args.evaluate == "dense":
            from src.retrieval.dense_retriever import DenseRetriever
            retriever = DenseRetriever(items)

            def dense_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                with collector.stage("dense") as st:
                    results = retriever.search(query_text, top_k)
                    st.output_ids = results
                collector.finalize_query(results, (time.perf_counter() - t0) * 1000)
                return results

            results = evaluate_retriever(dense_pipeline, queries, gt, k=args.k, collector=collector)

        elif args.evaluate == "hybrid":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)

            def hybrid_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                with collector.stage("bm25") as st:
                    bm25_results = bm25.search(query_text, top_k=BM25_TOP_K)
                    st.output_ids = bm25_results
                with collector.stage("dense") as st:
                    dense_results = dense.search(query_text, top_k=DENSE_TOP_K)
                    st.output_ids = dense_results
                with collector.stage("rrf_fusion") as st:
                    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=RRF_K, top_k=top_k)
                    st.output_ids = fused
                collector.finalize_query(fused, (time.perf_counter() - t0) * 1000)
                return fused

            results = evaluate_retriever(hybrid_pipeline, queries, gt, k=args.k, collector=collector)

        elif args.evaluate == "full":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
            from src.retrieval.llm_reranker import LLMReranker
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            reranker = LLMReranker(items)

            def full_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                with collector.stage("bm25") as st:
                    bm25_results = bm25.search(query_text, top_k=BM25_TOP_K)
                    st.output_ids = bm25_results
                with collector.stage("dense") as st:
                    dense_results = dense.search(query_text, top_k=DENSE_TOP_K)
                    st.output_ids = dense_results
                with collector.stage("rrf_fusion") as st:
                    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=RRF_K, top_k=RERANK_TOP_N)
                    st.output_ids = fused
                with collector.stage("llm_rerank") as st:
                    final = reranker.rerank(query_text, fused, top_k=top_k)
                    st.output_ids = final
                collector.finalize_query(final, (time.perf_counter() - t0) * 1000)
                return final

            results = evaluate_retriever(full_pipeline, queries, gt, k=args.k, collector=collector)

        elif args.evaluate == "hf":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
            from src.retrieval.llm_reranker import HFReranker
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            reranker = HFReranker(items)

            def hf_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                with collector.stage("bm25") as st:
                    bm25_results = bm25.search(query_text, top_k=BM25_TOP_K)
                    st.output_ids = bm25_results
                with collector.stage("dense") as st:
                    dense_results = dense.search(query_text, top_k=DENSE_TOP_K)
                    st.output_ids = dense_results
                with collector.stage("rrf_fusion") as st:
                    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=RRF_K, top_k=RERANK_TOP_N)
                    st.output_ids = fused
                with collector.stage("hf_rerank") as st:
                    final = reranker.rerank(query_text, fused, top_k=top_k)
                    st.output_ids = final
                collector.finalize_query(final, (time.perf_counter() - t0) * 1000)
                return final

            results = evaluate_retriever(hf_pipeline, queries, gt, k=args.k, collector=collector)

        print(json.dumps(results, indent=2, ensure_ascii=False))

        # Save plain metrics (backward compat)
        os.makedirs(EVAL_DIR, exist_ok=True)
        out_path = f"{EVAL_DIR}/eval_{args.evaluate}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {out_path}")

        # Save unified run report with traces
        config_snapshot = {
            "BM25_TOP_K": BM25_TOP_K, "DENSE_TOP_K": DENSE_TOP_K,
            "RRF_K": RRF_K, "RERANK_TOP_N": RERANK_TOP_N,
            "FINAL_TOP_K": FINAL_TOP_K,
            "EMBEDDING_MODEL": EMBEDDING_MODEL, "LLM_MODEL": LLM_MODEL,
        }
        report = build_run_report(args.evaluate, args.k, results, collector, config_snapshot)
        report_path = save_run_report(report, args.evaluate)
        print(f"Run report saved to {report_path}")


if __name__ == "__main__":
    main()
