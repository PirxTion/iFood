import hashlib
import os
import pickle
import numpy as np
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, DENSE_TOP_K, EMBEDDING_CACHE_PATH


def _cache_key(model: str, text: str) -> str:
    """Cache key that includes the model name so different models never collide."""
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


def _is_local_model(model: str) -> bool:
    """Local HuggingFace models contain a '/' (e.g. intfloat/multilingual-e5-small)."""
    return "/" in model


def _is_e5_model(model: str) -> bool:
    """E5 models need 'query: ' / 'passage: ' prefixes for best performance."""
    return "e5" in model.lower()


class DenseRetriever:
    def __init__(self, items: list[dict], model: str = EMBEDDING_MODEL, batch_size: int = 100):
        """Initialize dense retriever by embedding all items.

        Each item must have 'item_id' and 'text' keys.
        Embeddings are cached to disk so repeated runs skip recomputation.

        Args:
            items: list of dicts with 'item_id' and 'text' keys.
            model: embedding model identifier. Use 'org/name' for local
                   sentence-transformers models, or an OpenAI model name.
            batch_size: number of texts to embed per batch.
        """
        self.model = model
        self.item_ids = [item["item_id"] for item in items]

        if _is_local_model(model):
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model)
        else:
            from openai import OpenAI
            from src.config import PROXY_URL, PROXY_KEY
            key = PROXY_KEY or os.environ.get("PROXY_KEY", "")
            self._openai_client = OpenAI(api_key=key, base_url=PROXY_URL)

        texts = [item["text"] for item in items]
        self.item_embeddings = self._embed_batch(texts, batch_size)

    # ── cache ────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, list[float]]:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    # ── embedding helpers ────────────────────────────────────────────────

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        """Embed texts using a local sentence-transformers model."""
        return self._st_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def _embed_openai(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Embed texts using the OpenAI API in batches, returns list of vectors."""
        embeddings: list[list[float]] = []
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Embedding items (API)"):
            batch = texts[batch_start:batch_start + batch_size]
            resp = self._openai_client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(e.embedding for e in resp.data)
        return embeddings

    # ── batch & query embedding (with cache) ─────────────────────────────

    def _embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts, using disk cache to skip already-embedded texts."""
        cache = self._load_cache()

        if _is_e5_model(self.model):
            prefixed_texts = [f"passage: {t}" for t in texts]
        else:
            prefixed_texts = texts

        keys = [_cache_key(self.model, t) for t in prefixed_texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in cache]

        if missing_indices:
            missing_texts = [prefixed_texts[i] for i in missing_indices]
            if _is_local_model(self.model):
                vecs = self._embed_local(missing_texts)
                for j, idx in enumerate(missing_indices):
                    cache[keys[idx]] = vecs[j].tolist()
            else:
                vecs = self._embed_openai(missing_texts, batch_size)
                for j, idx in enumerate(missing_indices):
                    cache[keys[idx]] = vecs[j]
            self._save_cache(cache)

        return np.array([cache[k] for k in keys])

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (also cache it)."""
        if _is_e5_model(self.model):
            query = f"query: {query}"

        cache = self._load_cache()
        k = _cache_key(self.model, query)
        if k not in cache:
            if _is_local_model(self.model):
                vec = self._st_model.encode([query], normalize_embeddings=True)[0]
                cache[k] = vec.tolist()
            else:
                resp = self._openai_client.embeddings.create(model=self.model, input=[query])
                cache[k] = resp.data[0].embedding
            self._save_cache(cache)
        return np.array(cache[k])

    # ── search ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[str]:
        """Search by cosine similarity, return top_k item IDs."""
        query_emb = self._embed_query(query)

        norms = np.linalg.norm(self.item_embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = self.item_embeddings @ query_emb / (norms * query_norm + 1e-10)

        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.item_ids[i] for i in top_indices]
