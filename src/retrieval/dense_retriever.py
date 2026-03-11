import hashlib
import os
import pickle
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.config import PROXY_URL, PROXY_KEY, EMBEDDING_MODEL, DENSE_TOP_K, EMBEDDING_CACHE_PATH


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class DenseRetriever:
    def __init__(self, items: list[dict], batch_size: int = 100):
        """Initialize dense retriever by embedding all items.

        Each item must have 'item_id' and 'text' keys.
        Embeddings are cached to disk so repeated runs skip the API call.
        """
        key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
        self.client = OpenAI(api_key=key, base_url=PROXY_URL)
        self.item_ids = [item["item_id"] for item in items]

        texts = [item["text"] for item in items]
        self.item_embeddings = self._embed_batch(texts, batch_size)

    def _load_cache(self) -> dict[str, list[float]]:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    def _embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts in batches, using disk cache to skip already-embedded texts."""
        cache = self._load_cache()

        # Identify which texts need embedding
        hashes = [_text_hash(t) for t in texts]
        missing_indices = [i for i, h in enumerate(hashes) if h not in cache]

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            for batch_start in tqdm(range(0, len(missing_texts), batch_size), desc="Embedding items"):
                batch = missing_texts[batch_start:batch_start + batch_size]
                resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                for j, e in enumerate(resp.data):
                    h = _text_hash(batch[j])
                    cache[h] = e.embedding
            self._save_cache(cache)

        return np.array([cache[h] for h in hashes])

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (also cache it)."""
        cache = self._load_cache()
        h = _text_hash(query)
        if h not in cache:
            resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
            cache[h] = resp.data[0].embedding
            self._save_cache(cache)
        return np.array(cache[h])

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[str]:
        """Search by cosine similarity, return top_k item IDs."""
        query_emb = self._embed_query(query)

        norms = np.linalg.norm(self.item_embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = self.item_embeddings @ query_emb / (norms * query_norm + 1e-10)

        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.item_ids[i] for i in top_indices]
