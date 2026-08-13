"""Feature-family ablation experiment helpers (research-only)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd


def incremental_families(
    base_cols: Sequence[str],
    families: Dict[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """
    Build cumulative feature sets:
      baseline
      baseline + session
      baseline + session + regime
      ...
    """
    out: Dict[str, List[str]] = {"baseline": list(base_cols)}
    current = list(base_cols)
    for name, cols in families.items():
        current = list(dict.fromkeys(current + list(cols)))
        out[f"baseline+{name}"] = current
    return out


def summarize_family_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through helper for experiment notebooks."""
    return df.copy()
