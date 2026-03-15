# Query Router Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPT-4o-mini query router that classifies queries into R1/R2/R3 and dispatches each to its optimal retrieval strategy (BM25, dense, or dense-minus-negation + cross-encoder rerank).

**Architecture:** A new `QueryRouter` class makes a single structured JSON call to GPT-4o-mini and returns a `RouteResult` dataclass. A new `routed` evaluation mode in `run_eval.py` branches on the route: R1→BM25, R2→Dense, R3→Dense(main_term) minus BM25(negated_term) then CrossEncoderReranker. The tracing layer gets a `metadata` field to store the route decision.

**Tech Stack:** Python, openai SDK, pytest, existing `BM25Retriever`, `DenseRetriever`, `CrossEncoderReranker` from `src/retrieval/`.

**Spec:** `docs/superpowers/specs/2026-03-15-query-router-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/eval/tracing.py` | Modify | Add `metadata: dict` field to `StageTrace` and `StageTimer` |
| `src/config.py` | Modify | Add `ROUTER_MODEL = "gpt-4o-mini"` constant |
| `src/retrieval/query_router.py` | **Create** | `RouteResult` dataclass + `QueryRouter` class |
| `tests/test_query_router.py` | **Create** | Unit tests for `QueryRouter` (mocked OpenAI) |
| `tests/test_tracing.py` | Modify | Add tests for `metadata` field |
| `run_eval.py` | Modify | Add `"routed"` mode + `routed_pipeline` function |

---

## Chunk 1: Tracing metadata + config constant

### Task 1: Extend tracing with metadata support

**Files:**
- Modify: `tests/test_tracing.py`
- Modify: `src/eval/tracing.py`

- [ ] **Step 1: Add failing tests for metadata**

Append to `tests/test_tracing.py`:

```python
def test_stage_timer_forwards_metadata():
    qt = QueryTrace(query="pizza", category="keyword")
    with StageTimer("router", qt) as st:
        st.metadata = {"route": "R1", "main_term": None}
    assert qt.stages[0].metadata == {"route": "R1", "main_term": None}


def test_stage_trace_default_metadata_is_empty_dict():
    st = StageTrace(name="test")
    assert st.metadata == {}


def test_query_trace_serializes_metadata_to_json():
    import json
    qt = QueryTrace(query="pizza", category="keyword")
    qt.stages.append(StageTrace(name="router", time_ms=5.0, output_ids=[], metadata={"route": "R2"}))
    d = qt.to_dict()
    assert json.dumps(d)  # must be JSON-serializable
    assert d["stages"][0]["metadata"] == {"route": "R2"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tracing.py::test_stage_timer_forwards_metadata tests/test_tracing.py::test_stage_trace_default_metadata_is_empty_dict tests/test_tracing.py::test_query_trace_serializes_metadata_to_json -v
```

Expected: FAIL — `StageTrace.__init__() got an unexpected keyword argument 'metadata'`

- [ ] **Step 3: Add `metadata` field to `StageTrace` and wire through `StageTimer`**

In `src/eval/tracing.py`:

**`StageTrace`** — add the field (after `output_ids`):
```python
@dataclass
class StageTrace:
    name: str
    time_ms: float = 0.0
    output_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

**`StageTimer.__init__`** — add the attribute:
```python
def __init__(self, name: str, query_trace: QueryTrace):
    self.name = name
    self.query_trace = query_trace
    self.output_ids: list[str] = []
    self.metadata: dict = {}
    self._start: float = 0.0
```

**`StageTimer.__exit__`** — pass `metadata` when constructing `StageTrace`:
```python
def __exit__(self, *exc):
    elapsed_ms = (time.perf_counter() - self._start) * 1000
    self.query_trace.stages.append(
        StageTrace(name=self.name, time_ms=elapsed_ms, output_ids=self.output_ids, metadata=self.metadata)
    )
    return False
```

- [ ] **Step 4: Run all tracing tests to verify they pass**

```bash
uv run pytest tests/test_tracing.py -v
```

Expected: all PASS. The new `metadata` field has a default of `{}` so all existing tests continue to pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/eval/tracing.py tests/test_tracing.py
git commit -m "feat: add metadata field to StageTrace and StageTimer"
```

---

### Task 2: Add ROUTER_MODEL to config

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add the constant**

In `src/config.py`, add after `LLM_MODEL`:

```python
ROUTER_MODEL = "gpt-4o-mini"
```

- [ ] **Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: add ROUTER_MODEL config constant"
```

---

## Chunk 2: QueryRouter

### Task 3: Implement QueryRouter

**Files:**
- Create: `tests/test_query_router.py`
- Create: `src/retrieval/query_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_query_router.py`:

```python
# tests/test_query_router.py
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.query_router import QueryRouter, RouteResult


def _mock_response(content: str):
    """Build a minimal mock that looks like an OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("src.retrieval.query_router.OpenAI")
def test_classify_r1(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response('{"route": "R1"}')

    result = QueryRouter().classify("Pizza")

    assert result.route == "R1"
    assert result.main_term is None
    assert result.negated_term is None


@patch("src.retrieval.query_router.OpenAI")
def test_classify_r2(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response('{"route": "R2"}')

    result = QueryRouter().classify("Jantar romântico com massa")

    assert result.route == "R2"
    assert result.main_term is None
    assert result.negated_term is None


@patch("src.retrieval.query_router.OpenAI")
def test_classify_r3_extracts_terms(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response(
        '{"route": "R3", "main_term": "Macarrão", "negated_term": "frutos do mar"}'
    )

    result = QueryRouter().classify("Macarrão sem frutos do mar")

    assert result.route == "R3"
    assert result.main_term == "Macarrão"
    assert result.negated_term == "frutos do mar"


@patch("src.retrieval.query_router.OpenAI")
def test_classify_uses_correct_model_and_temperature(mock_cls):
    from src.config import ROUTER_MODEL

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response('{"route": "R1"}')

    QueryRouter().classify("Sushi")

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == ROUTER_MODEL
    assert kwargs["temperature"] == 0


@patch("src.retrieval.query_router.OpenAI")
def test_classify_falls_back_to_r2_on_malformed_json(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response("not valid json at all")

    result = QueryRouter().classify("some query")

    assert result.route == "R2"
    assert result.main_term is None
    assert result.negated_term is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_query_router.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.query_router'`

- [ ] **Step 3: Implement QueryRouter**

Create `src/retrieval/query_router.py`:

```python
# src/retrieval/query_router.py
import json
import os
from dataclasses import dataclass

from openai import OpenAI

from src.config import ROUTER_MODEL

_SYSTEM_PROMPT = """\
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

Respond with JSON only. No explanation.\
"""


@dataclass
class RouteResult:
    route: str                    # "R1", "R2", or "R3"
    main_term: str | None = None  # R3 only: positive search term
    negated_term: str | None = None  # R3 only: term to exclude


class QueryRouter:
    def __init__(self):
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def classify(self, query: str) -> RouteResult:
        """Classify a query into R1, R2, or R3 and extract terms for R3.

        Falls back to R2 (semantic/dense) if the LLM response cannot be parsed.
        """
        resp = self._client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            return RouteResult(
                route=data["route"],
                main_term=data.get("main_term"),
                negated_term=data.get("negated_term"),
            )
        except (json.JSONDecodeError, KeyError):
            return RouteResult(route="R2")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_query_router.py -v
```

Expected: all 5 PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/query_router.py tests/test_query_router.py
git commit -m "feat: implement QueryRouter with R1/R2/R3 classification"
```

---

## Chunk 3: Routed pipeline in run_eval.py

### Task 4: Wire routed_pipeline into run_eval.py

**Files:**
- Modify: `run_eval.py`

- [ ] **Step 1: Add `ROUTER_MODEL` to the config import block and add `import warnings` to stdlib imports**

In `run_eval.py`, find the existing stdlib imports at the top and add `warnings`:

```python
import argparse
import json
import os
import time
import warnings
```

Then add `ROUTER_MODEL` to the `from src.config import (...)` block (lines 12–15):

```python
from src.config import (
    EVAL_DIR, BM25_TOP_K, DENSE_TOP_K, RRF_K, RERANK_TOP_N,
    FINAL_TOP_K, EMBEDDING_MODEL, LLM_MODEL, CROSS_ENCODER_MODEL, ROUTER_MODEL,
)
```

- [ ] **Step 2: Add `"routed"` to the `--evaluate` choices**

Find the `parser.add_argument("--evaluate", choices=[...])` line and add `"routed"`:

```python
parser.add_argument(
    "--evaluate",
    choices=["bm25", "dense", "hybrid", "full", "hf", "dense_rerank", "routed"],
    help="Evaluate a retriever (full=hybrid+LLM reranker, hf=hybrid+CE reranker, "
         "dense_rerank=dense+CE reranker, routed=query-router-dispatched pipeline)",
)
```

- [ ] **Step 3: Add the `routed_pipeline` branch**

After the `elif args.evaluate == "dense_rerank":` block (around line 165), add:

```python
        elif args.evaluate == "routed":
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.dense_retriever import DenseRetriever
            from src.retrieval.llm_reranker import CrossEncoderReranker
            from src.retrieval.query_router import QueryRouter
            bm25 = BM25Retriever(items)
            dense = DenseRetriever(items)
            reranker = CrossEncoderReranker(items)
            router = QueryRouter()

            def routed_pipeline(query_text, top_k):
                cat = _query_category(queries, query_text)
                collector.begin_query(query_text, cat)
                t0 = time.perf_counter()

                with collector.stage("router") as st:
                    route_result = router.classify(query_text)
                    st.metadata = {
                        "route": route_result.route,
                        "main_term": route_result.main_term,
                        "negated_term": route_result.negated_term,
                    }

                if route_result.route == "R1":
                    with collector.stage("bm25") as st:
                        results = bm25.search(query_text, top_k)
                        st.output_ids = results

                elif route_result.route == "R2":
                    with collector.stage("dense") as st:
                        results = dense.search(query_text, top_k)
                        st.output_ids = results

                else:  # R3
                    # Guard: fall back to query_text if router omitted terms
                    main_term = route_result.main_term or query_text
                    negated_term = route_result.negated_term or ""

                    with collector.stage("dense_main") as st:
                        dense_candidates = dense.search(main_term, top_k=DENSE_TOP_K)
                        st.output_ids = dense_candidates

                    with collector.stage("bm25_negation") as st:
                        negation_ids = bm25.search(negated_term, top_k=BM25_TOP_K) if negated_term else []
                        st.output_ids = negation_ids

                    negation_set = set(negation_ids)
                    with collector.stage("negation_filter") as st:
                        filtered = [i for i in dense_candidates if i not in negation_set]
                        st.output_ids = filtered
                        if len(filtered) < top_k:
                            warnings.warn(
                                f"R3 negation filter: query='{query_text}', "
                                f"negated='{negated_term}', "
                                f"filtered count={len(filtered)} < top_k={top_k}"
                            )

                    with collector.stage("ce_rerank") as st:
                        results = reranker.rerank(query_text, filtered, top_k=top_k)
                        st.output_ids = results

                collector.finalize_query(results, (time.perf_counter() - t0) * 1000)
                return results

            results = evaluate_retriever(routed_pipeline, queries, gt, k=args.k, collector=collector)
```

- [ ] **Step 4: Add `"routed"` to the `reranker_model` dict and `ROUTER_MODEL` to `config_snapshot`**

Find the `reranker_model` dict (around line 196):

```python
reranker_model = {
    "full": LLM_MODEL,
    "hf": CROSS_ENCODER_MODEL,
    "dense_rerank": CROSS_ENCODER_MODEL,
    "routed": CROSS_ENCODER_MODEL,
}.get(args.evaluate)
```

Then, after the `config_snapshot` dict is built, add the router model conditionally:

```python
config_snapshot = {
    "BM25_TOP_K": BM25_TOP_K, "DENSE_TOP_K": DENSE_TOP_K,
    "RRF_K": RRF_K, "RERANK_TOP_N": RERANK_TOP_N,
    "FINAL_TOP_K": FINAL_TOP_K,
    "EMBEDDING_MODEL": EMBEDDING_MODEL, "LLM_MODEL": LLM_MODEL,
    "RERANKER_MODEL": reranker_model,
}
if args.evaluate == "routed":
    config_snapshot["ROUTER_MODEL"] = ROUTER_MODEL
```

- [ ] **Step 5: Run the full test suite to verify nothing is broken**

```bash
uv run pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 6: Smoke test the routed pipeline**

```bash
uv run python run_eval.py --evaluate routed --k 10 2>&1 | head -60
```

Expected: router classifies queries, R1/R2/R3 routes fire, metrics printed, run report saved to `eval_data/run_routed_<timestamp>.json`

- [ ] **Step 7: Commit**

```bash
git add run_eval.py
git commit -m "feat: add routed evaluation pipeline with query router dispatch"
```
