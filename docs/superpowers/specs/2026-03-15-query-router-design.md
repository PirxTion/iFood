# Query Router Design

Date: 2026-03-15

## Overview

Add a `QueryRouter` component that classifies each incoming search query into one of three routes and dispatches it to the appropriate retrieval strategy. This replaces the one-size-fits-all hybrid pipeline with a route-aware pipeline that uses the best retriever for each query type.

---

## Routes

| Route | Query type | Example | Retrieval strategy |
|-------|-----------|---------|-------------------|
| R1 | Keyword | "Pizza", "Sushi" | BM25 only |
| R2 | Semantic | "Jantar romântico com massa" | Dense only |
| R3 | Negative | "Macarrão sem frutos do mar" | Dense(main_term) − BM25(negated_term) |

---

## Components

### New: `src/retrieval/query_router.py`

**`RouteResult` dataclass**
```python
@dataclass
class RouteResult:
    route: str            # "R1", "R2", or "R3"
    main_term: str | None = None      # R3 only: positive search term
    negated_term: str | None = None   # R3 only: term to exclude
```

**`QueryRouter` class**
- Instantiates `OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))` — standard OpenAI API, no proxy.
- Note: `DenseRetriever` also reads `OPENAI_API_KEY`. Both use the same env var intentionally; they just target different API endpoints (embeddings vs. chat completions).
- Single public method: `classify(query: str) -> RouteResult`
- Makes one `gpt-4o-mini` call at `temperature=0` for determinism.
- Parses the JSON response into a `RouteResult`.

**System prompt:**
```
You are a query classifier for a Portuguese food delivery search system (iFood, Brazil).

Classify the query into exactly one route:
- R1 (keyword): Short, specific queries with no descriptive context.
  Examples: "Pizza", "Sushi", "X-Burguer"
- R2 (semantic): Descriptive or conceptual queries.
  Examples: "Jantar romântico com massa", "Comida saudável para almoço"
- R3 (negative): Queries that explicitly exclude an ingredient or attribute using
  "sem" or similar negation.
  Examples: "Macarrão sem frutos do mar", "Pizza sem glúten"

For R1 or R2, respond with JSON only:
{"route": "R1"} or {"route": "R2"}

For R3, also extract the main search term and the negated term:
{"route": "R3", "main_term": "Macarrão", "negated_term": "frutos do mar"}

Respond with JSON only. No explanation.
```

### Modified: `src/config.py`

Add:
```python
ROUTER_MODEL = "gpt-4o-mini"
```

### Modified: `run_eval.py`

- Add `"routed"` to the `--evaluate` argument choices.
- Add `routed_pipeline(query_text, top_k)` function (see Data Flow below).
- Import `ROUTER_MODEL` from `src.config` alongside the existing model imports.
- Include `ROUTER_MODEL` in the `config_snapshot` dict for the `routed` mode, so run reports are reproducible.

---

## Data Flow

```
query_text
    │
    ▼
QueryRouter.classify(query_text)
    │
    ├── R1 → bm25.search(query_text, top_k)
    │                                          ──► results
    ├── R2 → dense.search(query_text, top_k)
    │                                          ──► results
    └── R3 → dense.search(main_term, DENSE_TOP_K)   → dense_candidates
             bm25.search(negated_term, BM25_TOP_K)  → negation_set
             [id for id in dense_candidates
              if id not in set(negation_set)][:top_k] ──► results
```

**R3 detail:** `main_term` (e.g. `"Macarrão"`) — not the full query — is passed to the dense retriever so the embedding is not confused by the negation clause. The BM25 negation search finds items that explicitly mention the negated ingredient; these are hard-excluded from the dense candidates. No reranker is applied in the initial implementation.

**R3 empty-result fallback:** If the negation filter produces fewer than `top_k` results (e.g. the negated term is very common), return the truncated list as-is — do not error or fall back to unfiltered results. Log a warning with the query, negated term, and result count so the behaviour is visible in evaluation runs.

---

## Tracing Stages

| Route | Stages recorded |
|-------|----------------|
| R1 | `router` → `bm25` |
| R2 | `router` → `dense` |
| R3 | `router` → `dense_main` → `bm25_negation` → `negation_filter` |

The `router` stage records no `output_ids` (classification only). Route decision and extracted terms are stored in a `metadata` dict on the stage trace. This requires adding `metadata: dict = field(default_factory=dict)` to `StageTrace` in `src/eval/tracing.py`. The `StageTimer` context manager must also expose a `metadata` attribute so callers can write to it before `__exit__` appends the `StageTrace`.

Example for the `router` stage:
```python
with collector.stage("router") as st:
    result = router.classify(query_text)
    st.metadata = {"route": result.route, "main_term": result.main_term, "negated_term": result.negated_term}
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/retrieval/query_router.py` | **New** — `RouteResult`, `QueryRouter` |
| `src/config.py` | Add `ROUTER_MODEL` |
| `src/eval/tracing.py` | Add `metadata: dict = field(default_factory=dict)` to `StageTrace`; add `self.metadata: dict = {}` attribute to `StageTimer`; pass `metadata=self.metadata` in the `StageTrace(...)` constructor call inside `StageTimer.__exit__` |
| `run_eval.py` | Add `"routed"` mode and `routed_pipeline` |

---

## Out of Scope

- Reranking after R3 (deferred — try without first)
- Caching router results across eval runs
- Confidence scores or fallback routing
