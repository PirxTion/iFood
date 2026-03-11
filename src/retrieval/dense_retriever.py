import os
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.config import PROXY_URL, PROXY_KEY, EMBEDDING_MODEL, DENSE_TOP_K


class DenseRetriever:
    def __init__(self, items: list[dict], batch_size: int = 100):
        """Initialize dense retriever by embedding all items.

        Each item must have 'item_id' and 'text' keys.
        """
        key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
        self.client = OpenAI(api_key=key, base_url=PROXY_URL)
        self.item_ids = [item["item_id"] for item in items]

        texts = [item["text"] for item in items]
        self.item_embeddings = self._embed_batch(texts, batch_size)

    def _embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts in batches."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding items"):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_embs = [e.embedding for e in resp.data]
            all_embeddings.extend(batch_embs)
        return np.array(all_embeddings)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        return np.array(resp.data[0].embedding)

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[str]:
        """Search by cosine similarity, return top_k item IDs."""
        query_emb = self._embed_query(query)

        norms = np.linalg.norm(self.item_embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = self.item_embeddings @ query_emb / (norms * query_norm + 1e-10)

        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.item_ids[i] for i in top_indices]
