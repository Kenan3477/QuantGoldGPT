"""Immutable model registry stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ModelStage(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PAPER = "paper"
    PRODUCTION = "production"
    RETIRED = "retired"


@dataclass
class ModelRecord:
    model_id: str
    stage: str
    created_at: str
    git_commit: Optional[str]
    dataset_version: Optional[str]
    metrics: Dict[str, Any]
    artifact_path: str


class ModelRegistry:
    def __init__(self, root: str | Path = "artifacts/models"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if not self.index_path.exists():
            self._write([])

    def _read(self) -> list:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, rows: list) -> None:
        self.index_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def register(
        self,
        model_id: str,
        artifact_path: str,
        *,
        stage: ModelStage = ModelStage.CANDIDATE,
        git_commit: Optional[str] = None,
        dataset_version: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        rec = ModelRecord(
            model_id=model_id,
            stage=stage.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            dataset_version=dataset_version,
            metrics=metrics or {},
            artifact_path=artifact_path,
        )
        rows = self._read()
        rows.append(asdict(rec))
        self._write(rows)
        return rec

    def promote(self, model_id: str, stage: ModelStage) -> None:
        rows = self._read()
        found = False
        for row in rows:
            if row["model_id"] == model_id:
                row["stage"] = stage.value
                found = True
        if not found:
            raise KeyError(model_id)
        self._write(rows)
