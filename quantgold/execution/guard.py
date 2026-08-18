"""
Import boundary guard.

Live execution code must not import quantgold.research.
"""

from __future__ import annotations

import sys
from typing import Iterable


RESEARCH_PREFIX = "quantgold.research"


def assert_no_research_imports(loaded_modules: Iterable[str] | None = None) -> None:
    modules = loaded_modules if loaded_modules is not None else list(sys.modules)
    offenders = [m for m in modules if m == RESEARCH_PREFIX or m.startswith(RESEARCH_PREFIX + ".")]
    if offenders:
        raise RuntimeError(
            "Execution boundary violation: research modules loaded in execution context: "
            + ", ".join(sorted(offenders))
        )
