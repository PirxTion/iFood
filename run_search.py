"""Run interactive search queries."""
import argparse
import warnings

from src.data_loader import load_items
from src.config import DENSE_TOP_K, RERANK_TOP_N
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.llm_reranker import LLMReranker, CrossEncoderReranker
from src.retrieval.query_router import QueryRouter


def _routed_search(
    query_text: str, top_k: int, router, bm25, dense, reranker, item_texts,
) -> list[str]:
    """Run the routed pipeline: R1=BM25+CE, R2=Dense, R3=Dense+negation filter."""
    route_result = router.classify(query_text)

    if route_result.route == "R1":
        candidates = bm25.search(query_text, top_k=RERANK_TOP_N)
        return reranker.rerank(query_text, candidates, top_k=top_k)

    elif route_result.route == "R2":
        return dense.search(query_text, top_k=top_k)

    else:  # R3
        main_term = route_result.main_term or query_text
        negated_term = route_result.negated_term or ""

        candidates = dense.search(main_term, top_k=DENSE_TOP_K)

        if negated_term:
            neg_parts = [p.strip().lower() for p in negated_term.split(" e ") if p.strip()]
            filtered = [
                iid for iid in candidates
                if not any(part in item_texts[iid] for part in neg_parts)
            ]
        else:
            filtered = candidates

        if len(filtered) < top_k:
            warnings.warn(
                f"R3 negation filter: only {len(filtered)} items remain after filtering "
                f"'{negated_term}' (need {top_k})"
            )
        return filtered[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Semantic search over iFood items")
    parser.add_argument(
        "--mode",
        choices=["routed", "bm25", "dense", "hybrid", "full", "hf"],
        default="routed",
        help="Retrieval mode (routed=query-router pipeline [default], "
             "full=hybrid+LLM reranker, hf=hybrid+HF cross-encoder)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("query", nargs="?", help="Search query (omit for interactive mode)")
    args = parser.parse_args()

    print("Loading items...")
    items = load_items()
    item_lookup = {item["item_id"]: item for item in items}

    print(f"Initializing {args.mode} retriever...")
    bm25 = dense = hybrid = reranker = router = None
    item_texts = None

    if args.mode == "routed":
        bm25 = BM25Retriever(items)
        dense = DenseRetriever(items)
        reranker = CrossEncoderReranker(items)
        router = QueryRouter()
        item_texts = {
            item["item_id"]: (
                item["text"] + " " +
                " ".join(v for tag in item["tags"] for v in tag.get("value", []) if isinstance(v, str))
            ).lower()
            for item in items
        }
    else:
        if args.mode in ("bm25", "hybrid", "full", "hf"):
            bm25 = BM25Retriever(items)
        if args.mode in ("dense", "hybrid", "full", "hf"):
            dense = DenseRetriever(items)
        if args.mode in ("hybrid", "full", "hf"):
            hybrid = HybridRetriever(bm25, dense)
        if args.mode == "full":
            reranker = LLMReranker(items)
        elif args.mode == "hf":
            reranker = CrossEncoderReranker(items)

    def search(query_text: str) -> list[str]:
        if args.mode == "routed":
            return _routed_search(query_text, args.top_k, router, bm25, dense, reranker, item_texts)
        elif args.mode == "bm25":
            return bm25.search(query_text, args.top_k)
        elif args.mode == "dense":
            return dense.search(query_text, args.top_k)
        elif args.mode == "hybrid":
            return hybrid.search(query_text, args.top_k)
        else:  # full or hf
            candidates = hybrid.search(query_text, top_k=20)
            return reranker.rerank(query_text, candidates, top_k=args.top_k)

    def display_results(results: list[str]):
        for i, item_id in enumerate(results, 1):
            item = item_lookup.get(item_id, {})
            name = item.get("name", "?")
            cat = item.get("category_name", "?")
            price = item.get("price", 0)
            desc = item.get("description", "")[:60]
            print(f"  {i:2}. [{item_id}] {name} | {cat} | R${price:.2f}")
            if desc:
                print(f"       {desc}")

    if args.query:
        results = search(args.query)
        display_results(results)
    else:
        print(f"Interactive mode ({args.mode}). Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            results = search(query)
            display_results(results)
            print()


if __name__ == "__main__":
    main()
