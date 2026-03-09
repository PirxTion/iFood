# Semantic Search System for iFood — Design Document

Date: 2026-03-09

## Overview

A semantic search system that processes 60 Portuguese queries (semantic, keyword, hard negative) against 5,000 food/grocery items and returns ranked relevant results. Built in phases: evaluation pipeline first, then baseline retrieval, then optimizations.

## Data Summary

- **5,000 items** across 18 L0 taxonomy categories (includes non-food: PET, MODA, ELETRO, etc.)
- **60 queries** in Portuguese: 20 semantic, 20 keyword, 20 negative ("X sem Y")
- 96% of items have images; 4% do not
- Item text fields: name, category_name, description, taxonomy (l0/l1/l2), tags (lacFree, organic, vegan)
- Item behavioral signals: conversionRate, orderingRate, reorderRate, deliveryTime, total_orders

## Phase 1: Evaluation Pipeline

### Ground Truth Construction (Tournament-Style LLM Judging)

**Round 1 — Coarse filtering:**
- Split 5,000 items into 5 batches of 1,000
- Per batch per query: ask GPT-4o-mini to select top 15 most relevant items (IDs + brief reasoning)
- Run each batch 2 times (temperature > 0) to capture variance
- Union all selected items → ~100-150 candidates per query

**Round 2 — Fine-grained scoring:**
- Put all ~150 candidates for a query in a single LLM call (~9K tokens of item text, well within 128K context)
- Ask for graded relevance labels: 0=irrelevant, 1=marginally relevant, 2=relevant, 3=highly relevant
- For negative queries, also flag constraint violations explicitly
- Run 3 times per query, take median score per item

**Total API calls:** 60 queries × (5×2 + 3) = 780 calls to GPT-4o-mini

**Output:** `eval_data/ground_truth.json` mapping `query → [(itemId, relevance_grade, violation_flag)]`

### Evaluation Metrics

**Global metrics (all query types):**
- NDCG@10 (primary)
- MRR
- Precision@5, Precision@10

**Per-category breakdown:** all metrics split by semantic / keyword / negative

**Negative-specific metrics:**
- NCVR@10 (Negative Constraint Violation Rate): fraction of top-10 results that violate the negated concept
- Penalized NDCG@10: violating items receive score = -3 (same magnitude as max relevance) instead of 0

**Ablation comparisons:**
1. BM25 only
2. Dense retrieval only
3. Hybrid (BM25 + Dense + RRF)
4. Hybrid + LLM re-ranking (full pipeline)

## Phase 1: Baseline Retrieval Pipeline

### Item Text Representation

Combined text field per item for indexing:
```
{name} | {category_name} | {description} | {taxonomy.l0}/{taxonomy.l1}/{taxonomy.l2}
```
Boolean tags appended when True: `| sem lactose | vegano | orgânico`

### Stage 1: Dual Retrieval

**BM25 (sparse):**
- Index all 5,000 item text representations
- Portuguese-aware tokenization (lowercase + whitespace, optionally with stemming)
- Returns top-50 per query

**Dense embeddings:**
- Encode items and queries with multilingual embedding model
- Candidate: `text-embedding-3-small` via proxy or local `intfloat/multilingual-e5-base`
- Cosine similarity, returns top-50 per query

### Stage 2: Reciprocal Rank Fusion (RRF)

```
score(item) = Σ 1/(k + rank_i)  where k=60
```
Merge BM25 top-50 and dense top-50 → produce fused top-50

### Stage 3: LLM Re-ranking

- Take top-20 from RRF
- GPT-4o-mini re-ranks with relevance scores
- Explicit instruction: for negation queries, items must NOT contain excluded ingredients
- Returns final top-10

### Query Type Coverage

| Query Type | BM25 | Dense | LLM Re-ranker |
|---|---|---|---|
| Keyword ("pizza calabresa") | Strong | Good | Confirms |
| Semantic ("Almoço estilo havaiano") | Weak | Strong | Enhances |
| Negative ("macarrão sem peixe") | Finds matches but also violations | Ignores negation | Critical — understands "sem" |

## Phase 2: Optimizations (Post-Baseline)

### 2a. Query Router
- GPT-4o-mini classifies query → `{type, complexity}`
- Simple keyword → BM25 only (skip dense + re-ranking)
- Semantic → dense or hybrid
- Negative → always full pipeline with LLM re-ranking
- Reduces avg latency and cost

### 2b. Query Cache
- Embedding cache of seen queries + results
- New query: embed → nearest cached query → if similarity > 0.95, return cached results or skip to re-ranking
- Effective since many queries are thematically similar

### 2c. Category Pre-filtering
- Use existing L0/L1/L2 taxonomy as subcategories
- Query router outputs likely relevant L0 categories
- Retrieval searches only within those categories
- Shrinks search space from 5,000 → ~500-1,000 items

## Phase 3: Visual Component & Fine-tuning (Deferred)

Deferred until baseline metrics are established. Potential directions:
- CLIP embeddings for items with images
- LLM-generated image captions as additional text features
- Fine-tune embedding model or re-ranker on LLM-generated ground truth (contingent on data sufficiency analysis)

## Project Structure

```
prosus_assignment/
├── data/                          # CSV files (gitignored)
├── src/
│   ├── data_loader.py             # Parse CSVs, build item text representations
│   ├── config.py                  # Proxy URL, model names, hyperparams
│   ├── eval/
│   │   ├── build_ground_truth.py  # Tournament-style LLM judging
│   │   ├── metrics.py             # NDCG, MRR, P@K, NCVR, penalized NDCG
│   │   └── evaluate.py            # Run eval on any retriever
│   └── retrieval/
│       ├── bm25_retriever.py      # BM25 sparse retrieval
│       ├── dense_retriever.py     # Embedding-based dense retrieval
│       ├── hybrid_retriever.py    # RRF fusion
│       └── llm_reranker.py        # GPT-4o-mini re-ranking
├── notebooks/
│   └── exploration.ipynb          # Data exploration, result analysis
├── eval_data/                     # Generated ground truth (gitignored)
├── docs/plans/                    # Design & planning docs
├── key.md                         # Proxy key (gitignored)
└── requirements.txt
```

## Key Dependencies

- `rank_bm25` — BM25 retrieval
- `openai` — proxy API calls (embeddings + LLM)
- `numpy`, `pandas` — data processing
- `scikit-learn` — optional TF-IDF baseline
- `sentence-transformers` — optional local embedding models

## Decisions & Trade-offs

1. **Eval-first approach:** Build ground truth before retrieval pipeline so we can measure every incremental change.
2. **Text-only baseline:** Images deferred until we know the text-only ceiling. 96% image coverage means we can add it later without redesigning.
3. **LLM re-ranking at inference:** Adds latency (~1-2s) but critical for negation handling. Can be bypassed for simple keyword queries via router.
4. **Tournament judging with median-of-3:** Balances cost vs reliability. Single-call scoring of ~150 items avoids cross-batch calibration issues.
5. **Fine-tuning deferred:** 5,000 items and 60 queries is small. Risk of overfitting outweighs potential gains at this stage. Revisit after baseline.
