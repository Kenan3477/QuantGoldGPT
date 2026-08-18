"""Typed QuantGold settings (YAML-backed, config-driven)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

try:
    import yaml
except ImportError:  # pragma: no cover - optional until deps installed
    yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class InstrumentConfig:
    symbol: str
    asset_class: str = "precious_metal"
    enabled: bool = True


@dataclass
class TimeframeConfig:
    name: str
    minutes: int


@dataclass
class TripleBarrierConfig:
    """Research parameters — not production assumptions."""

    upper_atr_mult: float = 1.5
    lower_atr_mult: float = 1.0
    max_holding_bars: int = 12
    atr_period: int = 14
    same_bar_policy: str = "ambiguous"  # ambiguous | favor_upper | favor_lower | no_trade
    min_move_atr: float = 0.0


@dataclass
class ValidationConfig:
    train_years: int = 3
    validation_years: int = 1
    test_years: int = 1
    step_years: int = 1
    embargo_bars: int = 12
    purge_label_horizon_bars: int = 12
    final_holdout_start: Optional[str] = None  # ISO date; never tune against this


@dataclass
class DecisionConfig:
    """Selective trading thresholds — coverage vs precision is researched, not hard-coded as truth."""

    min_calibrated_probability: float = 0.78  # Sprint 2 final: 8.8% coverage, 75% win rate, Sharpe 0.717, PF 4.57
    max_model_disagreement: float = 0.20
    enable_meta_label: bool = True
    allow_no_trade: bool = True


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 5.0
    max_positions: int = 2
    max_portfolio_heat_pct: float = 3.0
    # Confidence must never unbounded-scale size
    max_confidence_size_multiplier: float = 1.25


@dataclass
class ExecutionCostConfig:
    spread_points: float = 25.0
    commission_per_lot: float = 0.0
    slippage_points: float = 5.0
    latency_ms: int = 150
    reject_probability: float = 0.0


@dataclass
class QuantGoldSettings:
    project_name: str = "QuantGold"
    instruments: List[InstrumentConfig] = field(
        default_factory=lambda: [
            InstrumentConfig("XAUUSD"),
            InstrumentConfig("XAGUSD"),
        ]
    )
    timeframes: List[TimeframeConfig] = field(
        default_factory=lambda: [
            TimeframeConfig("M1", 1),
            TimeframeConfig("M5", 5),
            TimeframeConfig("M15", 15),
            TimeframeConfig("H1", 60),
            TimeframeConfig("H4", 240),
            TimeframeConfig("D1", 1440),
        ]
    )
    primary_timeframe: str = "M15"
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    data_root: str = "artifacts/datasets"
    model_registry_root: str = "artifacts/models"
    experiment_root: str = "experiments"
    random_seed: int = 42

    def instrument_symbols(self) -> List[str]:
        return [i.symbol for i in self.instruments if i.enabled]


def _merge_dataclass(cls, data: Optional[Dict[str, Any]]):
    if not data:
        return cls()
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_settings(path: Optional[str | Path] = None) -> QuantGoldSettings:
    """Load settings from YAML, falling back to defaults."""
    cfg_path = Path(path or os.environ.get("QUANTGOLD_CONFIG", DEFAULT_CONFIG_PATH))
    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        if yaml is None:
            raise ImportError("PyYAML is required to load QuantGold YAML configs")
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    instruments = [
        _merge_dataclass(InstrumentConfig, item)
        for item in raw.get("instruments", [{"symbol": "XAUUSD"}, {"symbol": "XAGUSD"}])
    ]
    timeframes = [
        _merge_dataclass(TimeframeConfig, item)
        for item in raw.get(
            "timeframes",
            [
                {"name": "M1", "minutes": 1},
                {"name": "M5", "minutes": 5},
                {"name": "M15", "minutes": 15},
                {"name": "H1", "minutes": 60},
                {"name": "H4", "minutes": 240},
                {"name": "D1", "minutes": 1440},
            ],
        )
    ]

    return QuantGoldSettings(
        project_name=raw.get("project_name", "QuantGold"),
        instruments=instruments,
        timeframes=timeframes,
        primary_timeframe=raw.get("primary_timeframe", "M15"),
        triple_barrier=_merge_dataclass(TripleBarrierConfig, raw.get("triple_barrier")),
        validation=_merge_dataclass(ValidationConfig, raw.get("validation")),
        decision=_merge_dataclass(DecisionConfig, raw.get("decision")),
        risk=_merge_dataclass(RiskConfig, raw.get("risk")),
        costs=_merge_dataclass(ExecutionCostConfig, raw.get("costs")),
        data_root=raw.get("data_root", "artifacts/datasets"),
        model_registry_root=raw.get("model_registry_root", "artifacts/models"),
        experiment_root=raw.get("experiment_root", "experiments"),
        random_seed=int(raw.get("random_seed", 42)),
    )
