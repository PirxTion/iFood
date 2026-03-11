import time
import json
from src.eval.tracing import StageTrace, QueryTrace, StageTimer, QueryTraceCollector


def test_stage_timer_records_time_and_output():
    qt = QueryTrace(query="pizza", category="keyword")
    with StageTimer("bm25", qt) as st:
        st.output_ids = ["a", "b", "c"]
        time.sleep(0.01)
    assert len(qt.stages) == 1
    assert qt.stages[0].name == "bm25"
    assert qt.stages[0].time_ms >= 5  # at least ~10ms from sleep
    assert qt.stages[0].output_ids == ["a", "b", "c"]


def test_collector_begin_and_finalize():
    collector = QueryTraceCollector()
    qt = collector.begin_query("pizza", "keyword")
    with collector.stage("bm25") as st:
        st.output_ids = ["a"]
    collector.finalize_query(["a"], 100.0)
    assert len(collector.traces) == 1
    assert collector.traces[0].total_time_ms == 100.0
    assert collector.traces[0].final_ids == ["a"]


def test_collector_multiple_queries():
    collector = QueryTraceCollector()
    collector.begin_query("q1", "semantic")
    collector.finalize_query(["a"], 50.0)
    collector.begin_query("q2", "keyword")
    collector.finalize_query(["b"], 60.0)
    assert len(collector.traces) == 2


def test_set_query_metrics():
    collector = QueryTraceCollector()
    collector.begin_query("pizza", "keyword")
    collector.finalize_query(["a"], 50.0)
    collector.set_query_metrics("pizza", {"ndcg@10": 0.85})
    assert collector.traces[0].metrics == {"ndcg@10": 0.85}


def test_query_trace_serializes_to_json():
    qt = QueryTrace(query="pizza", category="keyword")
    qt.stages.append(StageTrace(name="bm25", time_ms=12.3, output_ids=["a", "b"]))
    qt.total_time_ms = 12.3
    qt.final_ids = ["a", "b"]
    qt.metrics = {"mrr": 1.0}
    d = qt.to_dict()
    assert json.dumps(d)  # must be JSON-serializable
    assert d["query"] == "pizza"
    assert d["stages"][0]["name"] == "bm25"
