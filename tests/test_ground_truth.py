# tests/test_ground_truth.py
import pytest
import json
from unittest.mock import patch, MagicMock
from src.eval.build_ground_truth import (
    format_items_for_prompt,
    parse_round1_response,
    build_round1_prompt,
)


def test_format_items_for_prompt():
    items = [
        {"item_id": "abc123", "name": "Pizza", "category_name": "Pizzas",
         "description": "Queijo e tomate", "taxonomy": {"l0": "A", "l1": "B", "l2": "C"},
         "price": 25.0},
        {"item_id": "def456", "name": "Sushi", "category_name": "Japonesa",
         "description": "Salmão", "taxonomy": {"l0": "A", "l1": "B", "l2": "C"},
         "price": 45.0},
    ]
    text = format_items_for_prompt(items)
    assert "abc123" in text
    assert "Pizza" in text
    assert "def456" in text
    assert "Sushi" in text


def test_parse_round1_response_json_list():
    response = '["abc123", "def456", "ghi789"]'
    result = parse_round1_response(response)
    assert result == ["abc123", "def456", "ghi789"]


def test_parse_round1_response_json_in_markdown():
    response = 'Here are the results:\n```json\n["abc123", "def456"]\n```'
    result = parse_round1_response(response)
    assert result == ["abc123", "def456"]


def test_build_round1_prompt_contains_query():
    prompt = build_round1_prompt("pizza calabresa", "item text block", top_n=15)
    assert "pizza calabresa" in prompt
    assert "15" in prompt
    assert "item text block" in prompt
