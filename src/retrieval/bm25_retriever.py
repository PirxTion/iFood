from rank_bm25 import BM25Plus

from src.config import BM25_TOP_K


class BM25Retriever:
    def __init__(self, items: list[dict]):
        """Initialize BM25 index from items.

        Each item must have 'item_id' and 'text' keys.
        """
        self.items = items
        self.item_ids = [item["item_id"] for item in items]

        # Tokenize: lowercase and split on whitespace/punctuation
        self.corpus = [self._tokenize(item["text"]) for item in items]
        self.bm25 = BM25Plus(self.corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return text.lower().split()

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[str]:
        """Search for query, return top_k item IDs."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.item_ids[i] for i in top_indices if scores[i] > 0][:top_k]
