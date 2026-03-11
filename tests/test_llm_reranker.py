# tests/test_llm_reranker.py
import pytest
from src.retrieval.llm_reranker import build_rerank_prompt, parse_rerank_response


def test_build_rerank_prompt_includes_query_and_items():
    prompt = build_rerank_prompt(
        query="pizza sem queijo",
        items_text="[id1] Pizza Margherita\n[id2] Pizza vegana",
        top_k=10,
    )
    assert "pizza sem queijo" in prompt
    assert "id1" in prompt
    assert "id2" in prompt
    assert "10" in prompt


def test_parse_rerank_response_json_list():
    response = '["id2", "id1", "id3"]'
    result = parse_rerank_response(response)
    assert result == ["id2", "id1", "id3"]


def test_parse_rerank_response_markdown():
    response = '```json\n["id2", "id1"]\n```'
    result = parse_rerank_response(response)
    assert result == ["id2", "id1"]
