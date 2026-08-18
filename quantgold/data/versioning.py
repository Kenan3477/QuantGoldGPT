"""Dataset version hashing for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def hash_dataframe(df: pd.DataFrame, cols: Optional[list[str]] = None) -> str:
    frame = df if cols is None else df[cols]
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def write_dataset_manifest(
    path: Path,
    *,
    symbol: str,
    timeframe: str,
    source: str,
    n_rows: int,
    start: str,
    end: str,
    content_hash: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest = {
        "symbol": symbol,
        "timeframe": timeframe,
        "source": source,
        "n_rows": n_rows,
        "start": start,
        "end": end,
        "content_hash": content_hash,
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def dataset_version_id(symbol: str, timeframe: str, content_hash: str) -> str:
    return f"{symbol.upper()}_{timeframe.upper()}_{content_hash}"
