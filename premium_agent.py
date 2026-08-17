from __future__ import annotations

import json
import os
from typing import Literal, Optional

from agno.agent import Agent
from agno.guardrails import PromptInjectionGuardrail
from agno.models.google import Gemini
from pydantic import BaseModel, Field

from data_tools import PremiumDataStore


Dimension = Literal["Client", "Treaty", "Portfolio", "Age", "Gender", "Quarter"]
Metric = Literal[
    "Overall Variance",
    "Change in mortality (COB1)",
    "Change in lapse (COB2)",
    "Variance Unexplained through COBS",
    "Accrual True Up",
    "LeftOver Persistency",
]


class QueryRoute(BaseModel):
    """LLM routing decision only. It must never contain calculated financial values."""

    intent: Literal["analysis", "comparison", "ranking", "overview", "out_of_scope", "unclear"]
    client: Optional[str] = None
    treaty: Optional[int] = None
    portfolio: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[str] = None
    quarter: Optional[str] = None
    quarter_1: Optional[str] = None
    quarter_2: Optional[str] = None
    group_by: Optional[Dimension] = None
    metric: Optional[Metric] = None
    top_n: Optional[int] = Field(default=None, ge=1, le=50)
    ranking_direction: Optional[Literal["ascending", "descending"]] = None
    reason: str


ROUTER_INSTRUCTIONS = """
You are the routing layer for a Premium Variance Analysis application.
Return ONLY the structured routing fields. Never answer the question and never calculate financial values.

The dataset dimensions are: Client, Treaty, Portfolio, Age, Gender, Quarter.
The approved metrics are:
- Overall Variance
- Change in mortality (COB1)
- Change in lapse (COB2)
- Variance Unexplained through COBS
- Accrual True Up
- LeftOver Persistency

Intents:
- analysis: filter or explain one slice of data, including driver questions.
- comparison: explicitly compare TWO named quarters.
- ranking: top/bottom/best/worst/highest/lowest requests.
- overview: explicit request for the complete dataset summary.
- out_of_scope: unrelated to the uploaded premium variance dataset.
- unclear: in-scope request that lacks enough concrete information to execute safely.

Extraction rules:
1. Extract filters exactly from the user's words. Do not invent a client, treaty, portfolio, age threshold, gender, or quarter.
2. Exact age: set age_min and age_max to the same number. Age range: set both bounds.
3. Do NOT infer numeric ages from vague phrases like 'young', 'older', or 'younger lives'; mark unclear unless the user provides a number/range.
4. For comparison, fill quarter_1 and quarter_2 in the order stated by the user. Do not also set quarter.
5. For ranking, set metric, group_by, top_n (default 10 if omitted), and ranking_direction.
6. 'Worst', 'lowest', 'most adverse', or 'most negative' means ascending. 'Best', 'highest', or 'most favourable' means descending.
7. 'Persistency' means the approved metric 'LeftOver Persistency'. 'COB1' means Change in mortality (COB1). 'COB2' means Change in lapse (COB2).
8. If the user asks to break down results by a dimension, put that dimension in group_by. Otherwise leave group_by null and Python will choose a safe default.
9. Questions such as 'Was persistency the main driver for Group in Q3?' are analysis requests with portfolio=Group, quarter=Q3, metric=LeftOver Persistency.
10. 'Across all quarters' is analysis with group_by=Quarter, not comparison, unless exactly two quarters are named.
11. Never output a number that came from the dataset. Your only job is routing.
"""

NARRATOR_INSTRUCTIONS = """
You are the management-commentary layer for a Premium Variance Analysis application.
You receive VERIFIED JSON produced by Python. The JSON is the only source of truth.

Mandatory rules:
1. Use only supplied values and statements. Never invent, alter, estimate, or recalculate a financial value.
2. Positive Overall Variance is Favourable; negative Overall Variance is Unfavourable.
3. Explain the main variance drivers using Change in mortality (COB1), Change in lapse (COB2), Accrual True Up, and LeftOver Persistency.
4. If Python says LeftOver Persistency is negative, you may say it MAY indicate higher lapse activity or lower retention. Never claim an observed lapse increase or invent a lapse rate.
5. If Python says persistency is the largest adverse driver, call that out explicitly.
6. For quarter comparisons, explain the direction of movement. A positive movement in Overall Variance is an improvement; a negative movement is a deterioration.
7. State any reconciliation problem if supplied. Do not say data reconciles unless the JSON says so.
8. Do not use external actuarial knowledge, web information, or unsupported causal explanations.
9. Keep management commentary concise: normally one or two short paragraphs.
10. You may format numbers with normal thousands separators in prose, but do not add a currency symbol because the dataset has no currency field.
11. If rows are truncated, do not imply the displayed rows represent the complete detailed population; use the totals for overall conclusions.
"""


class PremiumVarianceAgent:
    """Controlled pipeline: Gemini understands/narrates; Python owns the numbers."""

    def __init__(self, data_store: PremiumDataStore, model_id: str | None = None):
        self.data_store = data_store
        self.model_id = model_id or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        guardrail = PromptInjectionGuardrail()

        self.router = Agent(
            name="Premium Query Router",
            model=Gemini(id=self.model_id),
            output_schema=QueryRoute,
            instructions=ROUTER_INSTRUCTIONS,
            pre_hooks=[guardrail],
            markdown=False,
        )
        self.narrator = Agent(
            name="Premium Variance Narrator",
            model=Gemini(id=self.model_id),
            instructions=NARRATOR_INSTRUCTIONS,
            pre_hooks=[guardrail],
            markdown=True,
        )

    @staticmethod
    def _parse_route(content: object) -> QueryRoute:
        if isinstance(content, QueryRoute):
            return content
        if isinstance(content, dict):
            return QueryRoute.model_validate(content)
        return QueryRoute.model_validate_json(str(content))

    @staticmethod
    def _common_filters(route: QueryRoute) -> dict:
        return {
            "client": route.client,
            "treaty": route.treaty,
            "portfolio": route.portfolio,
            "age_min": route.age_min,
            "age_max": route.age_max,
            "gender": route.gender,
        }

    def _verified_result(self, route: QueryRoute) -> dict:
        if route.intent == "overview":
            return self.data_store.get_dataset_overview()

        if route.intent == "analysis":
            return self.data_store.analyse_data(
                **self._common_filters(route),
                quarter=route.quarter,
                group_by=route.group_by,
            )

        if route.intent == "comparison":
            if not route.quarter_1 or not route.quarter_2:
                return {
                    "status": "unclear",
                    "message": "Please specify two quarters to compare, for example Q1 and Q4.",
                    "rows": [],
                }
            return self.data_store.compare_quarters(
                route.quarter_1,
                route.quarter_2,
                **self._common_filters(route),
            )

        if route.intent == "ranking":
            if route.metric is None:
                return {
                    "status": "unclear",
                    "message": "Please specify which metric you want to rank.",
                    "rows": [],
                }
            return self.data_store.rank_results(
                metric=route.metric,
                group_by=route.group_by or "Treaty",
                direction=route.ranking_direction or "ascending",
                top_n=route.top_n or 10,
                **self._common_filters(route),
                quarter=route.quarter,
            )

        raise ValueError(f"Unsupported route intent: {route.intent}")

    @staticmethod
    def _narration_payload(verified: dict) -> dict:
        """Keep the LLM payload grounded and bounded."""
        payload = dict(verified)
        if isinstance(payload.get("rows"), list) and len(payload["rows"]) > 20:
            payload["rows"] = payload["rows"][:20]
            payload["narration_rows_limited"] = True
        if isinstance(payload.get("detail_rows"), list) and len(payload["detail_rows"]) > 20:
            payload["detail_rows"] = payload["detail_rows"][:20]
            payload["narration_detail_rows_limited"] = True
        return payload

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            return {
                "status": "unclear",
                "message": "Ask about a client, treaty, portfolio, age, gender, quarter, comparison, or ranking.",
                "rows": [],
            }

        try:
            route_output = self.router.run(question.strip())
        except Exception as exc:
            message = str(exc).lower()
            if "prompt injection" in message or "guardrail" in message:
                return {
                    "status": "blocked",
                    "message": "The request was blocked by the agent guardrails.",
                    "rows": [],
                }
            raise
        route = self._parse_route(route_output.content)

        if route.intent == "out_of_scope":
            return {
                "status": "out_of_scope",
                "message": "This agent can only analyse information contained in the uploaded Premium Variance dataset.",
                "rows": [],
                "route": route.model_dump(),
            }

        if route.intent == "unclear":
            return {
                "status": "unclear",
                "message": (
                    "I need a concrete dataset attribute or comparison. For example: 'Explain Term in Q3', "
                    "'Show age 45', or 'Compare Client 001 between Q1 and Q4'."
                ),
                "rows": [],
                "route": route.model_dump(),
            }

        verified = self._verified_result(route)
        if verified.get("status") != "success":
            return {**verified, "route": route.model_dump()}

        narration_input = (
            "Write management commentary using only this verified JSON:\n"
            + json.dumps(self._narration_payload(verified), indent=2, default=str)
        )
        try:
            narration_output = self.narrator.run(narration_input)
            commentary = str(narration_output.content)
            narration_error = None
        except Exception as exc:
            # Preserve the verified table even if the narration model has a temporary issue.
            commentary = (
                "Verified analysis completed successfully, but the LLM commentary could not be generated. "
                "The table above is still calculated deterministically from the workbook."
            )
            narration_error = str(exc)

        return {
            **verified,
            "commentary": commentary,
            "narration_error": narration_error,
            "route": route.model_dump(),
            "model": self.model_id,
        }
