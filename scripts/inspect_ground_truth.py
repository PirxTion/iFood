"""
Inspect ground-truth results by looking up item details from the dataset.

Usage:
    uv run scripts/inspect_ground_truth.py
    uv run scripts/inspect_ground_truth.py --query "açaí 500ml granola"
    uv run scripts/inspect_ground_truth.py --query "açaí 500ml granola" --max-items 5
    uv run scripts/inspect_ground_truth.py --category semantic
"""

import ast
import json
import argparse
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
GROUND_TRUTH_PATH = ROOT / "eval_data" / "ground_truth.json"
ITEMS_PATH = ROOT / "data" / "5k_items.csv"


def load_items(path: Path) -> dict[str, dict]:
    """Return a dict mapping itemId -> parsed metadata."""
    df = pd.read_csv(path)
    items = {}
    for _, row in df.iterrows():
        meta = ast.literal_eval(row["itemMetadata"])
        items[row["itemId"]] = {
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "category": meta.get("category_name", ""),
            "tags": [f"{t['key']}={','.join(t.get('value', []))}" for t in meta.get("tags", []) if isinstance(t, dict)],
            "vegan": meta.get("vegan", False),
            "organic": meta.get("organic", False),
            "lacFree": meta.get("lacFree", False),
            "taxonomy": meta.get("taxonomy", {}),
            "price": meta.get("price", None),
        }
    return items


def format_item(item_id: str, entry: dict, item_data: dict | None) -> str:
    relevance = entry["relevance"]
    violation = entry["violation"]
    stars = "★" * relevance + "☆" * (3 - relevance)

    if item_data is None:
        return f"  [{stars}] {item_id}  (not found in dataset)"

    tax = item_data["taxonomy"]
    taxonomy_str = " > ".join(filter(None, [tax.get("l0"), tax.get("l1"), tax.get("l2")]))
    tags_str = ", ".join(item_data["tags"][:5]) if item_data["tags"] else "—"
    flags = []
    if item_data["vegan"]:
        flags.append("vegan")
    if item_data["organic"]:
        flags.append("organic")
    if item_data["lacFree"]:
        flags.append("lac-free")
    flags_str = f"  [{', '.join(flags)}]" if flags else ""
    violation_str = "  ⚠ VIOLATION" if violation else ""

    name = item_data["name"] or "(no name)"
    desc = item_data["description"] or ""
    desc_short = textwrap.shorten(desc, width=80, placeholder="…")

    lines = [
        f"  [{stars}] {name}{violation_str}",
        f"         id: {item_id}",
        f"   category: {item_data['category']}",
        f"   taxonomy: {taxonomy_str}",
        f"       tags: {tags_str}{flags_str}",
    ]
    if desc_short:
        lines.append(f"       desc: {desc_short}")
    return "\n".join(lines)


def inspect(
    ground_truth: dict,
    items: dict,
    query_filter: str | None,
    category_filter: str | None,
    max_items: int,
):
    # Load queries to get categories if needed
    queries_df = None
    if category_filter:
        queries_df = pd.read_csv(ROOT / "data" / "queries.csv")
        # Normalise column names
        queries_df.columns = [c.strip().lower() for c in queries_df.columns]
        cat_col = next((c for c in queries_df.columns if "categ" in c or "type" in c), None)
        query_col = next((c for c in queries_df.columns if "query" in c or "search" in c or "term" in c or "text" in c or c == "q"), None)

    for query, entries in ground_truth.items():
        if query_filter and query_filter.lower() not in query.lower():
            continue

        if category_filter and queries_df is not None and cat_col and query_col:
            row = queries_df[queries_df[query_col].str.strip() == query]
            if row.empty or row.iloc[0][cat_col].lower() != category_filter.lower():
                continue

        shown = entries[:max_items]
        print(f"\n{'='*70}")
        print(f"Query: {query!r}  ({len(entries)} ground-truth items)")
        print(f"{'='*70}")

        for entry in shown:
            item_id = entry["item_id"]
            item_data = items.get(item_id)
            print(format_item(item_id, entry, item_data))

        if len(entries) > max_items:
            print(f"  ... and {len(entries) - max_items} more items (use --max-items to show more)")


def main():
    parser = argparse.ArgumentParser(description="Inspect ground-truth results with item details.")
    parser.add_argument("--query", "-q", default=None, help="Filter to queries containing this substring.")
    parser.add_argument("--category", "-c", default=None, choices=["semantic", "keyword", "hard_negative", "hard negative"],
                        help="Filter by query category.")
    parser.add_argument("--max-items", "-n", type=int, default=10,
                        help="Max items to show per query (default: 10).")
    args = parser.parse_args()

    print("Loading items…")
    items = load_items(ITEMS_PATH)
    print(f"  Loaded {len(items):,} items.")

    print("Loading ground truth…")
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)
    print(f"  Loaded {len(ground_truth)} queries.")

    inspect(ground_truth, items, args.query, args.category, args.max_items)
    print()


if __name__ == "__main__":
    main()
