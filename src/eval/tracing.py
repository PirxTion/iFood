"""Lightweight tracing for evaluation pipeline instrumentation."""
import time
from dataclasses import dataclass, field, asdict


@dataclass
class StageTrace:
    """Trace data for a single pipeline stage execution."""
    name: str
    time_ms: float = 0.0
    output_ids: list[str] = field(default_factory=list)


@dataclass
class QueryTrace:
    """Trace data for a single query through the full pipeline."""
    query: str
    category: str
    stages: list[StageTrace] = field(default_factory=list)
    total_time_ms: float = 0.0
    final_ids: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class StageTimer:
    """Context manager that times a stage and records its output.

    Usage:
        with StageTimer("bm25", query_trace) as st:
            results = bm25.search(query, top_k)
            st.output_ids = results
    """
    def __init__(self, name: str, query_trace: QueryTrace):
        self.name = name
        self.query_trace = query_trace
        self.output_ids: list[str] = []
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.query_trace.stages.append(
            StageTrace(name=self.name, time_ms=elapsed_ms, output_ids=self.output_ids)
        )
        return False


class QueryTraceCollector:
    """Collects trace data across all queries in an evaluation run."""

    def __init__(self):
        self.traces: list[QueryTrace] = []
        self._current: QueryTrace | None = None

    def begin_query(self, query: str, category: str) -> QueryTrace:
        qt = QueryTrace(query=query, category=category)
        self._current = qt
        self.traces.append(qt)
        return qt

    def stage(self, name: str) -> StageTimer:
        assert self._current is not None, "Call begin_query() first"
        return StageTimer(name, self._current)

    def finalize_query(self, final_ids: list[str], total_time_ms: float):
        assert self._current is not None
        self._current.final_ids = final_ids
        self._current.total_time_ms = total_time_ms

    def set_query_metrics(self, query_text: str, metrics: dict):
        for t in self.traces:
            if t.query == query_text:
                t.metrics = metrics
                return
