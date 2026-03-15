# tests/test_query_router.py
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.query_router import QueryRouter, RouteResult


def _mock_response(content: str):
    """Build a minimal mock that looks like an OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("src.retrieval.query_router.OpenAI")
def test_classify_r1(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response('{"route": "R1"}')

    result = QueryRouter().classify("Pizza")

    assert result.route == "R1"
    assert result.main_term is None
    assert result.negated_term is None


@patch("src.retrieval.query_router.OpenAI")
def test_classify_r2(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response('{"route": "R2"}')

    result = QueryRouter().classify("Jantar romântico com massa")

    assert result.route == "R2"
    assert result.main_term is None
    assert result.negated_term is None


@patch("src.retrieval.query_router.OpenAI")
def test_classify_r3_extracts_terms(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response(
        '{"route": "R3", "main_term": "Macarrão", "negated_term": "frutos do mar"}'
    )

    result = QueryRouter().classify("Macarrão sem frutos do mar")

    assert result.route == "R3"
    assert result.main_term == "Macarrão"
    assert result.negated_term == "frutos do mar"


@patch("src.retrieval.query_router.OpenAI")
def test_classify_uses_correct_model_and_temperature(mock_cls):
    from src.config import ROUTER_MODEL

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response('{"route": "R1"}')

    QueryRouter().classify("Sushi")

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == ROUTER_MODEL
    assert kwargs["temperature"] == 0


@patch("src.retrieval.query_router.OpenAI")
def test_classify_falls_back_to_r2_on_malformed_json(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_response("not valid json at all")

    result = QueryRouter().classify("some query")

    assert result.route == "R2"
    assert result.main_term is None
    assert result.negated_term is None
