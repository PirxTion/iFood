"""Generate ground truth and/or evaluate retrievers."""
import argparse
import json
import os
import time
import warnings

from src.data_loader import load_items, load_queries
from src.eval.build_ground_truth import build_ground_truth, save_ground_truth, load_ground_truth
from src.eval.evaluate import evaluate_retriever
from src.eval.tracing import QueryTraceCollector
from src.eval.run_report import build_run_report, save_run_report
from src.config import (
    EVAL_DIR, BM25_TOP_K, DENSE_TOP_K, RRF_K, RERANK_TOP_N,
    FINAL_TOP_K, EMBEDDING_MODEL, LLM_MODEL, CROSS_ENCODER_MODEL, ROUTER_MODEL,
)


def _query_category(queries, query_text):
    """Look up the category for a query text."""
    return next((q["category"] for q in queries if q["query"] == query_text), "unknown")


def main():
    parser = argparse.ArgumentParser(description="Build ground truth or evaluate retrievers")
    parser.add_argument("--build-gt", action="store_true", help="Build ground truth via LLM")
    parser.add_argument(
        "--evaluate",
        choices=["bm25", "dense", "hybrid", "full", "hf", "dense_rerank", "routed"],
        help="Evaluate a retriever (full=hybrid+LLM reranker, hf=hybrid+CE reranker, "
             "dense_rerank=dense+CE reranker, routed=query-router-dispatched pipeline)",
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
            from src.retrieval.llm_reranker import CrossEncoderReranker
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            reranker = CrossEncoderReranker(items)

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

        elif args.evaluate == "dense_rerank":
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.llm_reranker import CrossEncoderReranker
            dense = DenseRetriever(items)
            reranker = CrossEncoderReranker(items)

            def dense_rerank_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                with collector.stage("dense") as st:
                    dense_results = dense.search(query_text, top_k=RERANK_TOP_N)
                    st.output_ids = dense_results
                with collector.stage("ce_rerank") as st:
                    final = reranker.rerank(query_text, dense_results, top_k=top_k)
                    st.output_ids = final
                collector.finalize_query(final, (time.perf_counter() - t0) * 1000)
                return final

            results = evaluate_retriever(dense_rerank_pipeline, queries, gt, k=args.k, collector=collector)

        elif args.evaluate == "routed":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.llm_reranker import CrossEncoderReranker
            from src.retrieval.query_router import QueryRouter
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            reranker = CrossEncoderReranker(items)
            router = QueryRouter()

            def routed_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()
                results: list[str] = []

                with collector.stage("router") as st:
                    route_result = router.classify(query_text)
                    st.metadata = {
                        "route": route_result.route,
                        "main_term": route_result.main_term,
                        "negated_term": route_result.negated_term,
                    }

                if route_result.route == "R1":
                    with collector.stage("bm25") as st:
                        results = bm25.search(query_text, top_k)
                        st.output_ids = results

                elif route_result.route == "R2":
                    with collector.stage("dense") as st:
                        results = dense.search(query_text, top_k)
                        st.output_ids = results

                else:  # R3
                    # Guard: fall back to query_text if router omitted terms
                    main_term = route_result.main_term or query_text
                    negated_term = route_result.negated_term or ""

                    with collector.stage("dense_main") as st:
                        dense_candidates = dense.search(main_term, top_k=DENSE_TOP_K)
                        st.output_ids = dense_candidates

                    with collector.stage("bm25_negation") as st:
                        negation_ids = bm25.search(negated_term, top_k=BM25_TOP_K) if negated_term else []
                        st.output_ids = negation_ids

                    negation_set = set(negation_ids)
                    with collector.stage("negation_filter") as st:
                        filtered = [i for i in dense_candidates if i not in negation_set]
                        st.output_ids = filtered
                        if len(filtered) < top_k:
                            warnings.warn(
                                f"R3 negation filter: query='{query_text}', "
                                f"negated='{negated_term}', "
                                f"filtered count={len(filtered)} < top_k={top_k}"
                            )

                    with collector.stage("ce_rerank") as st:
                        results = reranker.rerank(query_text, filtered, top_k=top_k)
                        st.output_ids = results

                collector.finalize_query(results, (time.perf_counter() - t0) * 1000)
                return results

            results = evaluate_retriever(routed_pipeline, queries, gt, k=args.k, collector=collector)

        print(json.dumps(results, indent=2, ensure_ascii=False))

        # Save plain metrics (backward compat)
        os.makedirs(EVAL_DIR, exist_ok=True)
        out_path = f"{EVAL_DIR}/eval_{args.evaluate}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {out_path}")

        # Save unified run report with traces
        reranker_model = {
            "full": LLM_MODEL,
            "hf": CROSS_ENCODER_MODEL,
            "dense_rerank": CROSS_ENCODER_MODEL,
            "routed": CROSS_ENCODER_MODEL,
        }.get(args.evaluate)
        config_snapshot = {
            "BM25_TOP_K": BM25_TOP_K, "DENSE_TOP_K": DENSE_TOP_K,
            "RRF_K": RRF_K, "RERANK_TOP_N": RERANK_TOP_N,
            "FINAL_TOP_K": FINAL_TOP_K,
            "EMBEDDING_MODEL": EMBEDDING_MODEL, "LLM_MODEL": LLM_MODEL,
            "RERANKER_MODEL": reranker_model,
        }
        if args.evaluate == "routed":
            config_snapshot["ROUTER_MODEL"] = ROUTER_MODEL
        report = build_run_report(args.evaluate, args.k, results, collector, config_snapshot)
        report_path = save_run_report(report, args.evaluate)
        print(f"Run report saved to {report_path}")


if __name__ == "__main__":
    main()
