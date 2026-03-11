import json
import os
from src.eval.run_report import build_timing_summary, build_run_report, save_run_report
from src.eval.tracing import QueryTraceCollector


def test_build_timing_summary():
    traces = [
        {
            "total_time_ms": 100.0,
            "stages": [
                {"name": "bm25", "time_ms": 10.0},
                {"name": "dense", "time_ms": 90.0},
            ],
        },
        {
            "total_time_ms": 200.0,
            "stages": [
                {"name": "bm25", "time_ms": 20.0},
                {"name": "dense", "time_ms": 180.0},
            ],
        },
    ]
    summary = build_timing_summary(traces)
    assert summary["total_seconds"] == 0.3
    assert summary["per_query_mean_seconds"] == 0.15
    assert "p90_seconds" in summary
    assert "p99_seconds" in summary
    assert summary["p90_seconds"] >= summary["per_query_mean_seconds"]
    assert summary["p99_seconds"] >= summary["p90_seconds"]
    assert summary["stage_breakdown"]["bm25"]["total_ms"] == 30.0
    assert summary["stage_breakdown"]["bm25"]["mean_ms"] == 15.0
    assert summary["stage_breakdown"]["bm25"]["max_ms"] == 20.0
    assert summary["stage_breakdown"]["bm25"]["min_ms"] == 10.0
    assert summary["stage_breakdown"]["bm25"]["count"] == 2


def test_build_timing_summary_empty():
    assert build_timing_summary([]) == {}


def test_build_run_report_structure():
    collector = QueryTraceCollector()
    collector.begin_query("pizza", "keyword")
    collector.finalize_query(["a"], 50.0)
    report = build_run_report(
        mode="bm25", k=10, metrics={"overall": {}}, collector=collector
    )
    assert "run_metadata" in report
    assert report["run_metadata"]["mode"] == "bm25"
    assert "metrics" in report
    assert "timing" in report
    assert "query_traces" in report
    assert len(report["query_traces"]) == 1


def test_save_run_report_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.eval.run_report.EVAL_DIR", str(tmp_path))
    report = {"run_metadata": {"mode": "test"}, "metrics": {}, "timing": {}, "query_traces": []}
    path = save_run_report(report, "test")
    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["run_metadata"]["mode"] == "test"
