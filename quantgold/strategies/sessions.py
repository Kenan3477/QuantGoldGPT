"""Session name helper shared by routing / features."""

from __future__ import annotations

from datetime import datetime


def session_from_hour_utc(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 16:
        return "overlap"
    if 16 <= hour < 21:
        return "new_york"
    return "off"


def session_from_timestamp(ts: datetime) -> str:
    return session_from_hour_utc(ts.hour)
