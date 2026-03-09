# src/data_loader.py
import ast
import csv
from typing import Any

from src.config import ITEMS_CSV, QUERIES_CSV


def build_item_text(
    name: str,
    category_name: str,
    description: str,
    taxonomy: dict,
    lac_free: bool = False,
    vegan: bool = False,
    organic: bool = False,
) -> str:
    """Build combined text representation for an item."""
    tax_str = f"{taxonomy.get('l0', '')}/{taxonomy.get('l1', '')}/{taxonomy.get('l2', '')}"
    parts = [name, category_name, description, tax_str]

    if lac_free:
        parts.append("sem lactose")
    if vegan:
        parts.append("vegano")
    if organic:
        parts.append("orgânico")

    return " | ".join(parts)


def load_items(path: str = ITEMS_CSV) -> list[dict[str, Any]]:
    """Load and parse items CSV into structured dicts."""
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = ast.literal_eval(row["itemMetadata"])
            profile = ast.literal_eval(row["itemProfile"])

            item = {
                "item_id": row["itemId"],
                "merchant_id": row["merchantId"],
                "name": meta["name"],
                "category_name": meta["category_name"],
                "description": meta["description"],
                "price": meta["price"],
                "images": meta["images"],
                "taxonomy": meta["taxonomy"],
                "lac_free": meta.get("lacFree", False),
                "vegan": meta.get("vegan", False),
                "organic": meta.get("organic", False),
                "tags": meta.get("tags", []),
                "metrics": profile.get("metrics", {}),
            }
            item["text"] = build_item_text(
                name=item["name"],
                category_name=item["category_name"],
                description=item["description"],
                taxonomy=item["taxonomy"],
                lac_free=item["lac_free"],
                vegan=item["vegan"],
                organic=item["organic"],
            )
            items.append(item)
    return items


def load_queries(path: str = QUERIES_CSV) -> list[dict[str, str]]:
    """Load queries CSV."""
    queries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query": row["search_term_pt"],
                "category": row["category"],
            })
    return queries
