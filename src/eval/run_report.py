"""Assemble and save unified run reports with traces and metrics."""
import json
import os
from datetime import datetime, timezone

from src.config import EVAL_DIR


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) via linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def build_timing_summary(traces: list[dict]) -> dict:
    """Compute aggregate timing stats from query trace dicts."""
    if not traces:
        return {}

    e2e_times_ms = [t["total_time_ms"] for t in traces]
    total_seconds = sum(e2e_times_ms) / 1000
    per_query_mean = total_seconds / len(traces)

    stage_times: dict[str, list[float]] = {}
    for t in traces:
        for s in t["stages"]:
            stage_times.setdefault(s["name"], []).append(s["time_ms"])

    stage_breakdown = {}
    for name, times in stage_times.items():
        stage_breakdown[name] = {
            "total_ms": sum(times),
            "mean_ms": sum(times) / len(times),
            "max_ms": max(times),
            "min_ms": min(times),
            "count": len(times),
        }

    return {
        "total_seconds": total_seconds,
        "per_query_mean_seconds": per_query_mean,
        "p90_seconds": _percentile(e2e_times_ms, 90) / 1000,
        "p99_seconds": _percentile(e2e_times_ms, 99) / 1000,
        "stage_breakdown": stage_breakdown,
    }


def build_run_report(
    mode: str,
    k: int,
    metrics: dict,
    collector,  # QueryTraceCollector
    config_snapshot: dict | None = None,
) -> dict:
    """Assemble the full run report."""
    trace_dicts = [t.to_dict() for t in collector.traces]
    return {
        "run_metadata": {
            "mode": mode,
            "k": k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_snapshot or {},
        },
        "metrics": metrics,
        "timing": build_timing_summary(trace_dicts),
        "query_traces": trace_dicts,
    }


def save_run_report(report: dict, mode: str) -> str:
    """Save report to eval_data/ and return the path."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{EVAL_DIR}/run_{mode}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
