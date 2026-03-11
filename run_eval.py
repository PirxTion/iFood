"""Generate ground truth and/or evaluate retrievers."""
import argparse
import json
import os

from src.data_loader import load_items, load_queries
from src.eval.build_ground_truth import build_ground_truth, save_ground_truth, load_ground_truth
from src.eval.evaluate import evaluate_retriever
from src.config import EVAL_DIR


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

        if args.evaluate == "bm25":
            from src.retrieval.bm25_retriever import BM25Retriever
            retriever = BM25Retriever(items)
            results = evaluate_retriever(retriever.search, queries, gt, k=args.k)

        elif args.evaluate == "dense":
            from src.retrieval.dense_retriever import DenseRetriever
            retriever = DenseRetriever(items)
            results = evaluate_retriever(retriever.search, queries, gt, k=args.k)

        elif args.evaluate == "hybrid":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            hybrid = HybridRetriever(bm25, dense)
            results = evaluate_retriever(hybrid.search, queries, gt, k=args.k)

        elif args.evaluate == "full":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.retrieval.llm_reranker import LLMReranker
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            hybrid = HybridRetriever(bm25, dense)
            reranker = LLMReranker(items)

            def full_pipeline(query_text, top_k):
                candidates = hybrid.search(query_text, top_k=20)
                return reranker.rerank(query_text, candidates, top_k=top_k)

            results = evaluate_retriever(full_pipeline, queries, gt, k=args.k)

        elif args.evaluate == "hf":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.retrieval.llm_reranker import HFReranker
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            hybrid = HybridRetriever(bm25, dense)
            reranker = HFReranker(items)

            def hf_pipeline(query_text, top_k):
                candidates = hybrid.search(query_text, top_k=20)
                return reranker.rerank(query_text, candidates, top_k=top_k)

            results = evaluate_retriever(hf_pipeline, queries, gt, k=args.k)

        print(json.dumps(results, indent=2, ensure_ascii=False))

        os.makedirs(EVAL_DIR, exist_ok=True)
        out_path = f"{EVAL_DIR}/eval_{args.evaluate}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
