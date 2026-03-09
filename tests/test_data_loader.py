# tests/test_data_loader.py
import pytest
from src.data_loader import load_items, load_queries, build_item_text


def test_load_items_returns_list_of_dicts():
    items = load_items()
    assert len(items) == 5000
    first = items[0]
    assert "item_id" in first
    assert "name" in first
    assert "category_name" in first
    assert "description" in first
    assert "taxonomy" in first
    assert "text" in first  # combined text representation


def test_load_items_parses_metadata():
    items = load_items()
    first = items[0]
    assert isinstance(first["taxonomy"], dict)
    assert "l0" in first["taxonomy"]
    assert isinstance(first["price"], float)
    assert isinstance(first["images"], list)


def test_load_queries_returns_list_of_dicts():
    queries = load_queries()
    assert len(queries) == 59
    first = queries[0]
    assert "query" in first
    assert "category" in first
    assert first["category"] in ("semantic", "keyword", "negative")


def test_build_item_text_includes_all_fields():
    text = build_item_text(
        name="Pizza Margherita",
        category_name="Pizzas",
        description="Massa fina com queijo",
        taxonomy={"l0": "ALIMENTOS_PREPARADOS", "l1": "PIZZAS", "l2": "PIZZA_TRADICIONAL"},
        lac_free=True,
        vegan=False,
        organic=False,
    )
    assert "Pizza Margherita" in text
    assert "Pizzas" in text
    assert "Massa fina com queijo" in text
    assert "ALIMENTOS_PREPARADOS" in text
    assert "sem lactose" in text
    assert "vegano" not in text


def test_build_item_text_no_tags_when_false():
    text = build_item_text(
        name="Test",
        category_name="Cat",
        description="Desc",
        taxonomy={"l0": "A", "l1": "B", "l2": "C"},
        lac_free=False,
        vegan=False,
        organic=False,
    )
    assert "sem lactose" not in text
    assert "vegano" not in text
    assert "orgânico" not in text
