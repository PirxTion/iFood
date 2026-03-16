# src/retrieval/query_router.py
import json
import os
from dataclasses import dataclass

from openai import OpenAI

from src.config import ROUTER_MODEL, OPENAI_API_KEY

_SYSTEM_PROMPT = """\
You are a query classifier for a Portuguese food delivery search system (iFood, Brazil).

Classify the query into exactly one route:
- R1 (keyword): Short, specific queries with no descriptive context.
  Examples: "Pizza", "Sushi", "X-Burguer"
- R2 (semantic): Descriptive or conceptual queries.
  Examples: "Jantar romântico com massa", "Comida saudável para almoço"
- R3 (negative): Queries that explicitly exclude an ingredient or attribute using
  "sem" or similar negation.
  Examples: "Macarrão sem frutos do mar", "Pizza sem glúten"

For R1 or R2, respond with JSON only:
{"route": "R1"} or {"route": "R2"}

For R3, also extract the main search term and the negated term:
{"route": "R3", "main_term": "Macarrão", "negated_term": "frutos do mar"}

Respond with JSON only. No explanation.\
"""


@dataclass
class RouteResult:
    route: str                    # "R1", "R2", or "R3"
    main_term: str | None = None  # R3 only: positive search term
    negated_term: str | None = None  # R3 only: term to exclude


class QueryRouter:
    def __init__(self):
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def classify(self, query: str) -> RouteResult:
        """Classify a query into R1, R2, or R3 and extract terms for R3.

        Falls back to R2 (semantic/dense) if the LLM response cannot be parsed.
        """
        resp = self._client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            return RouteResult(
                route=data["route"],
                main_term=data.get("main_term"),
                negated_term=data.get("negated_term"),
            )
        except (json.JSONDecodeError, KeyError):
            return RouteResult(route="R2")
