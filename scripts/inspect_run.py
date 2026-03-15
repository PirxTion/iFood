"""
Inspect a retrieval run report by expanding item IDs into human-readable item details.

Usage:
    uv run scripts/inspect_run.py eval_data/run_dense_20260315_155138.json
    uv run scripts/inspect_run.py eval_data/run_hybrid_20260315_155255.json --query "macarrão sem peixe"
    uv run scripts/inspect_run.py eval_data/run_bm25_20260315_155025.json --category negative
    uv run scripts/inspect_run.py eval_data/run_dense_20260315_155138.json --stages  # show per-stage IDs too
"""

import ast
import json
import argparse
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
ITEMS_PATH = ROOT / "data" / "5k_items.csv"
GROUND_TRUTH_PATH = ROOT / "eval_data" / "ground_truth.json"


def load_items(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path)
    items = {}
    for _, row in df.iterrows():
        meta = ast.literal_eval(row["itemMetadata"])
        items[row["itemId"]] = {
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "category": meta.get("category_name", ""),
            "vegan": meta.get("vegan", False),
            "organic": meta.get("organic", False),
            "lacFree": meta.get("lacFree", False),
            "taxonomy": meta.get("taxonomy", {}),
            "price": meta.get("price", None),
            "tags": [t.get("key", "") for t in meta.get("tags", []) if isinstance(t, dict)],
        }
    return items


def format_item(
    rank: int,
    item_id: str,
    item_data: dict | None,
    gt_entry: dict | None,
) -> str:
    rel_str = ""
    viol_str = ""
    if gt_entry is not None:
        rel = gt_entry.get("relevance", 0)
        rel_str = f"  rel={'★'*rel + '☆'*(3-rel)}"
        if gt_entry.get("violation"):
            viol_str = "  ⚠ VIOLATION"

    if item_data is None:
        return f"  #{rank:>2}  {item_id}  (not in dataset){rel_str}{viol_str}"

    tax = item_data["taxonomy"]
    taxonomy_str = " > ".join(filter(None, [tax.get("l0"), tax.get("l1"), tax.get("l2")]))
    flags = [f for f, v in [("vegan", item_data["vegan"]), ("organic", item_data["organic"]), ("lac-free", item_data["lacFree"])] if v]
    flags_str = f"  [{', '.join(flags)}]" if flags else ""
    price_str = f"  R${item_data['price']:.2f}" if item_data["price"] is not None else ""
    desc = textwrap.shorten(item_data["description"] or "", width=72, placeholder="…")

    lines = [
        f"  #{rank:>2}  {item_data['name']}{rel_str}{viol_str}{flags_str}",
        f"        id: {item_id}",
        f"  category: {item_data['category']}{price_str}",
        f"  taxonomy: {taxonomy_str}",
    ]
    if desc:
        lines.append(f"      desc: {desc}")
    return "\n".join(lines)


def inspect_run(
    run: dict,
    items: dict,
    ground_truth: dict,
    query_filter: str | None,
    category_filter: str | None,
    show_stages: bool,
):
    mode = run["run_metadata"]["mode"]
    overall = run["metrics"]["overall"]
    print(f"\nRun mode: {mode.upper()}   ndcg@10={overall.get('ndcg@10', 0):.3f}  "
          f"f1@10={overall.get('f1@10', 0):.3f}  mrr={overall.get('mrr', 0):.3f}")
    print(f"{'='*70}")

    for trace in run["query_traces"]:
        query = trace["query"]
        category = trace["category"]

        if query_filter and query_filter.lower() not in query.lower():
            continue
        if category_filter and category.lower() != category_filter.lower():
            continue

        metrics = trace.get("metrics", {})
        ndcg = metrics.get("ndcg@10", 0)
        mrr_val = metrics.get("mrr", 0)
        f1 = metrics.get("f1@10", 0)
        total_ms = trace.get("total_time_ms", 0)
        gt_lookup = {e["item_id"]: e for e in ground_truth.get(query, [])}

        print(f"\nQuery : {query!r}  [{category}]")
        print(f"Scores: ndcg@10={ndcg:.3f}  mrr={mrr_val:.3f}  f1@10={f1:.3f}  ({total_ms:.1f} ms)")

        if show_stages:
            for stage in trace.get("stages", []):
                print(f"\n  -- Stage: {stage['name']} ({stage['time_ms']:.1f} ms, {len(stage['output_ids'])} items) --")
                for rank, item_id in enumerate(stage["output_ids"], 1):
                    gt_entry = gt_lookup.get(item_id)
                    print(format_item(rank, item_id, items.get(item_id), gt_entry))
        else:
            print(f"\n  -- Final results ({len(trace['final_ids'])} items) --")
            for rank, item_id in enumerate(trace["final_ids"], 1):
                gt_entry = gt_lookup.get(item_id)
                print(format_item(rank, item_id, items.get(item_id), gt_entry))

        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect a retrieval run report with item details.")
    parser.add_argument("run_file", help="Path to run JSON file (e.g. eval_data/run_dense_*.json)")
    parser.add_argument("--query", "-q", default=None, help="Filter to queries containing this substring.")
    parser.add_argument("--category", "-c", default=None,
                        choices=["semantic", "keyword", "negative"],
                        help="Filter by query category.")
    parser.add_argument("--stages", action="store_true",
                        help="Show per-stage intermediate results (bm25, dense, rrf_fusion, etc.).")
    args = parser.parse_args()

    run_path = Path(args.run_file)
    if not run_path.is_absolute():
        run_path = ROOT / run_path

    print("Loading items…")
    items = load_items(ITEMS_PATH)
    print(f"  Loaded {len(items):,} items.")

    print("Loading ground truth…")
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    print(f"Loading run: {run_path.name}…")
    with open(run_path) as f:
        run = json.load(f)

    inspect_run(run, items, ground_truth, args.query, args.category, args.stages)


if __name__ == "__main__":
    main()
