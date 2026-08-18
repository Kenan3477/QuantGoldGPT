"""Prepare leakage-safe research datasets: features + triple-barrier labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from quantgold.config.settings import QuantGoldSettings, load_settings
from quantgold.data.store import CanonicalDataStore
from quantgold.features.bundle import FeatureBundle, FeatureBundleConfig, BuiltFeatures
from quantgold.features.registry import FORBIDDEN_LABEL_COLUMNS
from quantgold.labels.triple_barrier import LABEL_AMBIGUOUS, TripleBarrierLabeler


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    feature_columns: List[str]
    label_column: str
    symbol: str
    timeframe: str
    families: Dict[str, List[str]]
    dataset_version: str


def prepare_research_dataset(
    symbol: str,
    timeframe: str,
    *,
    settings: Optional[QuantGoldSettings] = None,
    store: Optional[CanonicalDataStore] = None,
    feature_config: Optional[FeatureBundleConfig] = None,
    externals: Optional[Dict[str, pd.DataFrame]] = None,
    peer_metal: Optional[pd.DataFrame] = None,
    events: Optional[pd.DataFrame] = None,
    drop_ambiguous: bool = True,
) -> PreparedDataset:
    settings = settings or load_settings()
    store = store or CanonicalDataStore(settings.data_root)
    raw = store.load_ohlcv(symbol, timeframe)

    built: BuiltFeatures = FeatureBundle(feature_config).transform(
        raw,
        externals=externals,
        peer_metal=peer_metal,
        events=events,
    )
    frame = built.frame
    tb = TripleBarrierLabeler(settings.triple_barrier).label(frame)
    frame["tb_label"] = tb.labels
    frame["tb_upper"] = tb.upper_barrier
    frame["tb_lower"] = tb.lower_barrier
    frame["tb_touch_bar"] = tb.touch_bar

    feature_columns = list(built.feature_columns)
    # Hard guarantee
    feature_columns = [c for c in feature_columns if c not in FORBIDDEN_LABEL_COLUMNS]
    for c in feature_columns:
        if c in FORBIDDEN_LABEL_COLUMNS:
            raise RuntimeError(f"Label leakage: {c}")

    # Optional sparse families (intermarket/macro) may be entirely NaN without externals.
    # Fill those; require non-null only for core causal families.
    optional_families = set(built.families.get("intermarket", [])) | set(
        built.families.get("macro", [])
    )
    core_cols = [c for c in feature_columns if c not in optional_families]
    for c in optional_families:
        if c in frame.columns:
            frame[c] = frame[c].fillna(0.0)

    usable = frame.dropna(subset=core_cols + ["tb_label"]).copy()
    if drop_ambiguous:
        usable = usable[usable["tb_label"] != LABEL_AMBIGUOUS]

    # Keep timeouts as class 0 (no clear directional event)
    version = f"{symbol}_{timeframe}_{len(usable)}"
    return PreparedDataset(
        frame=usable.reset_index(drop=True),
        feature_columns=feature_columns,
        label_column="tb_label",
        symbol=symbol.upper(),
        timeframe=timeframe.upper(),
        families=built.families,
        dataset_version=version,
    )
