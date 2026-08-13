"""Configuration-driven specialist model routing (stubs for M4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class RouteContext:
    instrument: str
    session: str
    regime: str
    volatility_bucket: str
    event_state: str = "normal"  # normal | pre_event | post_event | blocked


class SpecialistRouter:
    """
    Decide which specialist model keys are eligible.

    Does not invent edge — only routes. Eligibility lists are config-driven.
    """

    def __init__(self, routes: Optional[dict] = None):
        # Default illustrative map — research must validate each specialist.
        self.routes = routes or {
            ("XAUUSD", "london", "TRENDING_UP"): ["xau_london_continuation"],
            ("XAUUSD", "london", "RANGING"): ["xau_range_reversion"],
            ("XAUUSD", "new_york", "TRENDING_UP"): ["xau_ny_continuation"],
            ("XAUUSD", "new_york", "TRENDING_DOWN"): ["xau_ny_reversal"],
            ("XAGUSD", "london", "RANGING"): ["xag_range_reversion"],
            ("XAGUSD", "new_york", "TRENDING_DOWN"): ["xag_ny_reversal"],
        }

    def eligible_models(self, ctx: RouteContext) -> List[str]:
        if ctx.event_state == "blocked":
            return []
        key = (ctx.instrument.upper(), ctx.session.lower(), ctx.regime)
        models = list(self.routes.get(key, []))
        # Fallback generic baseline always eligible unless blocked
        if not models:
            models = [f"{ctx.instrument.lower()}_baseline"]
        return models
