import os

PROXY_URL = "https://oovault.nl/api/proxy/v1"
PROXY_KEY = os.environ.get("PROXY_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

DATA_DIR = "data"
EVAL_DIR = "eval_data"

ITEMS_CSV = f"{DATA_DIR}/5k_items.csv"
QUERIES_CSV = f"{DATA_DIR}/queries.csv"

# Retrieval hyperparams
BM25_TOP_K = 50
DENSE_TOP_K = 50
RRF_K = 60
RERANK_TOP_N = 20
FINAL_TOP_K = 10

# Eval ground truth params
GT_BATCH_SIZE = 1000
GT_ROUND1_TOP_PER_BATCH = 15
GT_ROUND1_RUNS = 2
GT_ROUND2_RUNS = 3
