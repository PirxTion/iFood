"""Run interactive search queries."""
import argparse

from src.data_loader import load_items
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.llm_reranker import LLMReranker, CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(description="Semantic search over iFood items")
    parser.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid", "full", "hf"],
        default="full",
        help="Retrieval mode (full=hybrid+LLM reranker, hf=hybrid+HF cross-encoder)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("query", nargs="?", help="Search query (omit for interactive mode)")
    args = parser.parse_args()

    print("Loading items...")
    items = load_items()
    item_lookup = {item["item_id"]: item for item in items}

    print(f"Initializing {args.mode} retriever...")
    bm25 = dense = hybrid = reranker = None

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
        if args.mode == "bm25":
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
