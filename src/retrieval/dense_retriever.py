import hashlib
import os
import pickle
import numpy as np
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM, DENSE_TOP_K, EMBEDDING_CACHE_PATH, PROXY_KEY


def _cache_key(model: str, text: str, dim: int | None = None, prompt: str | None = None) -> str:
    """Cache key that includes the model name, dim, and prompt so different configs never collide."""
    prefix = model
    if dim:
        prefix += f":d{dim}"
    if prompt:
        prefix += f":p={prompt}"
    return hashlib.sha256(f"{prefix}:{text}".encode()).hexdigest()


def _is_local_model(model: str) -> bool:
    """Local HuggingFace models contain a '/' (e.g. intfloat/multilingual-e5-small)."""
    return "/" in model


def _is_e5_model(model: str) -> bool:
    """E5 models need 'query: ' / 'passage: ' prefixes for best performance."""
    return "e5" in model.lower()


class DenseRetriever:
    def __init__(self, items: list[dict], model: str = EMBEDDING_MODEL, batch_size: int = 100,
                 dim: int | None = EMBEDDING_DIM):
        """Initialize dense retriever by embedding all items.

        Each item must have 'item_id' and 'text' keys.
        Embeddings are cached to disk so repeated runs skip recomputation.

        Args:
            items: list of dicts with 'item_id' and 'text' keys.
            model: embedding model identifier. Use 'org/name' for local
                   sentence-transformers models, or an OpenAI model name.
            batch_size: number of texts to embed per batch.
            dim: output dimension override (OpenAI models only). None = default.
        """
        self.model = model
        self.dim = dim if not _is_local_model(model) else None
        self.item_ids = [item["item_id"] for item in items]

        if _is_local_model(model):
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model)
            # Use built-in prompt templates if the model provides them
            self._doc_prompt = "document" if "document" in self._st_model.prompts else None
            self._query_prompt = "query" if "query" in self._st_model.prompts else None
        else:
            from openai import OpenAI
            key = PROXY_KEY or os.environ.get("OPENAI_API_KEY", "")
            self._openai_client = OpenAI(api_key=key)

        texts = [item["text"] for item in items]
        self.item_embeddings = self._embed_batch(texts, batch_size)

        # Normalise once for cosine similarity via dot product
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._normed = (self.item_embeddings / norms).astype(np.float32)

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

    def _embed_local(self, texts: list[str], prompt_name: str | None = None) -> np.ndarray:
        """Embed texts using a local sentence-transformers model."""
        kwargs = {"show_progress_bar": True, "normalize_embeddings": True}
        # Use model's built-in prompt templates if available (e.g. EmbeddingGemma)
        if prompt_name and prompt_name in self._st_model.prompts:
            kwargs["prompt_name"] = prompt_name
        return self._st_model.encode(texts, **kwargs)

    def _embed_openai(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Embed texts using the OpenAI API in batches, returns list of vectors."""
        embeddings: list[list[float]] = []
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Embedding items (API)"):
            batch = texts[batch_start:batch_start + batch_size]
            kwargs = {"model": self.model, "input": batch}
            if self.dim:
                kwargs["dimensions"] = self.dim
            resp = self._openai_client.embeddings.create(**kwargs)
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

        prompt_name = getattr(self, "_doc_prompt", None)
        keys = [_cache_key(self.model, t, self.dim, prompt_name) for t in prefixed_texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in cache]

        if missing_indices:
            missing_texts = [prefixed_texts[i] for i in missing_indices]
            if _is_local_model(self.model):
                vecs = self._embed_local(missing_texts, prompt_name=prompt_name)
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

        prompt_name = getattr(self, "_query_prompt", None)
        cache = self._load_cache()
        k = _cache_key(self.model, query, self.dim, prompt_name)
        if k not in cache:
            if _is_local_model(self.model):
                kwargs = {"normalize_embeddings": True}
                if prompt_name and prompt_name in self._st_model.prompts:
                    kwargs["prompt_name"] = prompt_name
                vec = self._st_model.encode([query], **kwargs)[0]
                cache[k] = vec.tolist()
            else:
                kwargs = {"model": self.model, "input": [query]}
                if self.dim:
                    kwargs["dimensions"] = self.dim
                resp = self._openai_client.embeddings.create(**kwargs)
                cache[k] = resp.data[0].embedding
            self._save_cache(cache)
        return np.array(cache[k])

    # ── search ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[str]:
        """Brute-force cosine search — fast enough for 5k items."""
        qvec = self._embed_query(query).astype(np.float32).reshape(1, -1)
        qvec /= (np.linalg.norm(qvec) or 1)
        scores = (self._normed @ qvec.T).squeeze()
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [self.item_ids[i] for i in top_indices]
