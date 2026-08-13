"""Lightweight experiment tracking (MLflow-compatible records without hard dependency)."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


@dataclass
class ExperimentRecord:
    experiment_id: str
    git_commit: Optional[str]
    dataset_version: str
    instrument: str
    timeframe: str
    features: list
    label_definition: Dict[str, Any]
    train_period: str
    validation_period: str
    test_period: str
    model: list
    hyperparameters: Dict[str, Any]
    threshold: float
    results: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExperimentTracker:
    def __init__(self, root: str | Path = "experiments"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentRecord) -> Path:
        path = self.root / f"{record.experiment_id}.json"
        path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
        # append index
        index = self.root / "index.jsonl"
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"experiment_id": record.experiment_id, "path": str(path)}) + "\n")
        return path

    def start(
        self,
        *,
        dataset_version: str,
        instrument: str,
        timeframe: str,
        features: list,
        label_definition: Dict[str, Any],
        model: list,
        hyperparameters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.65,
        train_period: str = "",
        validation_period: str = "",
        test_period: str = "",
        results: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRecord:
        rec = ExperimentRecord(
            experiment_id=str(uuid.uuid4()),
            git_commit=_git_commit(),
            dataset_version=dataset_version,
            instrument=instrument,
            timeframe=timeframe,
            features=features,
            label_definition=label_definition,
            train_period=train_period,
            validation_period=validation_period,
            test_period=test_period,
            model=model,
            hyperparameters=hyperparameters or {},
            threshold=threshold,
            results=results or {},
        )
        self.log(rec)
        return rec
