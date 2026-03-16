import os

PROXY_URL = "https://oovault.nl/api/proxy/v1"
PROXY_KEY = os.environ.get("PROXY_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBEDDING_MODEL = "models/embeddinggemma-finetuned-2"
EMBEDDING_DIM = None  # None = model default
LLM_MODEL = "gpt-4o-mini"
ROUTER_MODEL = "gpt-4o-mini"

DATA_DIR = "data"
EVAL_DIR = "eval_data"
EMBEDDING_CACHE_PATH = f"{EVAL_DIR}/embedding_cache.pkl"

ITEMS_CSV = f"{DATA_DIR}/5k_items.csv"
QUERIES_CSV = f"{DATA_DIR}/queries.csv"

# Retrieval hyperparams
BM25_TOP_K = 50
DENSE_TOP_K = 50
NEGATION_DENSE_TOP_K = 200  # Semantic exclusion top-K for R3 negation filtering
RRF_K = 60
RERANK_TOP_N = 20
FINAL_TOP_K = 10

# Eval ground truth params
GT_BATCH_SIZE = 500
GT_ROUND1_TOP_PER_BATCH = 10
GT_ROUND1_RUNS = 2
GT_ROUND2_RUNS = 3
