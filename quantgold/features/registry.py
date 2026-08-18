"""Feature registry and leakage guards."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

from quantgold.labels.triple_barrier import TripleBarrierLabeler

# Explicit denylist — extended from XAUBot leakage findings.
FORBIDDEN_LABEL_COLUMNS: Set[str] = set(TripleBarrierLabeler.label_columns()) | {
    "future_close",
    "_future_close",
    "target_return",
    "forward_return",
    "realized_pnl",
}


class FeatureRegistry:
    """Tracks approved feature names and rejects label leakage."""

    def __init__(self, approved: Sequence[str] | None = None):
        self._approved: Set[str] = set(approved or [])

    def register(self, names: Iterable[str]) -> None:
        for name in names:
            if name in FORBIDDEN_LABEL_COLUMNS:
                raise ValueError(f"Refusing to register label/leakage column as feature: {name}")
            self._approved.add(name)

    @property
    def approved(self) -> List[str]:
        return sorted(self._approved)

    def select(self, columns: Sequence[str]) -> List[str]:
        """Return intersection of columns with approved set, excluding forbidden."""
        out = []
        for c in columns:
            if c in FORBIDDEN_LABEL_COLUMNS:
                continue
            if self._approved and c not in self._approved:
                continue
            out.append(c)
        return out

    @staticmethod
    def assert_no_label_leakage(feature_columns: Sequence[str]) -> None:
        leaked = [c for c in feature_columns if c in FORBIDDEN_LABEL_COLUMNS]
        if leaked:
            raise ValueError(f"Label/leakage columns present in features: {leaked}")
