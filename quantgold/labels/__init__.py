"""Event-based labelling (triple-barrier and variants)."""

from quantgold.labels.triple_barrier import (
    TripleBarrierLabeler,
    TripleBarrierResult,
    LABEL_UP,
    LABEL_DOWN,
    LABEL_TIMEOUT,
    LABEL_AMBIGUOUS,
)

__all__ = [
    "TripleBarrierLabeler",
    "TripleBarrierResult",
    "LABEL_UP",
    "LABEL_DOWN",
    "LABEL_TIMEOUT",
    "LABEL_AMBIGUOUS",
]
