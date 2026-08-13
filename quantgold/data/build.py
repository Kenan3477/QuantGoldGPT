"""Build and persist canonical QuantGold datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from quantgold.config.settings import QuantGoldSettings, load_settings
from quantgold.data.ingest.base import MarketDataSource
from quantgold.data.ingest.synthetic import SyntheticSource
from quantgold.data.ingest.yfinance_source import YFinanceSource
from quantgold.data.store import CanonicalDataStore
from quantgold.data.timestamps import bar_available_timestamp
from quantgold.data.versioning import (
    dataset_version_id,
    hash_dataframe,
    write_dataset_manifest,
)

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


@dataclass
class BuiltDataset:
    symbol: str
    timeframe: str
    path: Path
    version_id: str
    n_rows: int
    source: str


def get_source(name: str = "yfinance", **kwargs) -> MarketDataSource:
    if name == "yfinance":
        return YFinanceSource()
    if name == "synthetic":
        return SyntheticSource(seed=int(kwargs.get("seed", 42)))
    if name == "mt5":
        from quantgold.data.ingest.mt5_source import MT5Source

        return MT5Source(**kwargs)
    raise ValueError(f"Unknown source: {name}")


def build_canonical_dataset(
    symbol: str,
    timeframe: str,
    *,
    source: MarketDataSource,
    store: Optional[CanonicalDataStore] = None,
    settings: Optional[QuantGoldSettings] = None,
    limit: Optional[int] = None,
) -> BuiltDataset:
    settings = settings or load_settings()
    store = store or CanonicalDataStore(settings.data_root)
    tf = timeframe.upper()
    minutes = TF_MINUTES[tf]

    raw = source.fetch_ohlcv(symbol, tf, limit=limit)
    raw["available_timestamp"] = raw["timestamp"].map(lambda ts: bar_available_timestamp(ts, minutes))
    path = store.save_ohlcv(raw, symbol, tf, minutes)
    df = store.load_ohlcv(symbol, tf)
    content_hash = hash_dataframe(df, ["timestamp", "open", "high", "low", "close", "volume"])
    version_id = dataset_version_id(symbol, tf, content_hash)
    write_dataset_manifest(
        path.with_suffix(".manifest.json"),
        symbol=symbol.upper(),
        timeframe=tf,
        source=source.name,
        n_rows=len(df),
        start=str(df["timestamp"].iloc[0]),
        end=str(df["timestamp"].iloc[-1]),
        content_hash=content_hash,
        extra={"version_id": version_id},
    )
    return BuiltDataset(symbol.upper(), tf, path, version_id, len(df), source.name)


def build_all_datasets(
    *,
    source_name: str = "yfinance",
    symbols: Optional[Iterable[str]] = None,
    timeframes: Optional[Iterable[str]] = None,
    settings: Optional[QuantGoldSettings] = None,
    limit: Optional[int] = None,
) -> List[BuiltDataset]:
    settings = settings or load_settings()
    source = get_source(source_name)
    symbols = list(symbols or settings.instrument_symbols())
    # Prefer research-friendly TFs by default (Yahoo constraints on M1)
    timeframes = list(timeframes or ["M15", "H1", "H4", "D1"])
    built: List[BuiltDataset] = []
    errors: List[str] = []
    for symbol in symbols:
        for tf in timeframes:
            try:
                built.append(
                    build_canonical_dataset(
                        symbol,
                        tf,
                        source=source,
                        settings=settings,
                        limit=limit,
                    )
                )
            except Exception as exc:  # pragma: no cover - network variability
                errors.append(f"{symbol}/{tf}: {exc}")
    if not built and errors:
        raise RuntimeError("Dataset build failed:\n" + "\n".join(errors))
    return built
