"""Compose feature families into a single matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from quantgold.features.base import BaseFeatureBuilder
from quantgold.features.intermarket import IntermarketFeatureBuilder
from quantgold.features.macro import MacroEventFeatureBuilder
from quantgold.features.registry import FeatureRegistry
from quantgold.features.sessions import SessionFeatureBuilder
from quantgold.features.structure import StructureFeatureBuilder


@dataclass
class FeatureBundleConfig:
    use_base: bool = True
    use_sessions: bool = True
    use_structure: bool = True
    use_intermarket: bool = True
    use_macro: bool = True


@dataclass
class BuiltFeatures:
    frame: pd.DataFrame
    feature_columns: List[str]
    families: Dict[str, List[str]] = field(default_factory=dict)


class FeatureBundle:
    def __init__(self, config: Optional[FeatureBundleConfig] = None):
        self.config = config or FeatureBundleConfig()
        self.base = BaseFeatureBuilder()
        self.sessions = SessionFeatureBuilder()
        self.structure = StructureFeatureBuilder()
        self.intermarket = IntermarketFeatureBuilder()
        self.macro = MacroEventFeatureBuilder()

    def transform(
        self,
        df: pd.DataFrame,
        *,
        externals: Optional[Dict[str, pd.DataFrame]] = None,
        peer_metal: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> BuiltFeatures:
        out = df.copy()
        families: Dict[str, List[str]] = {}
        cols: List[str] = []

        if self.config.use_base:
            fm = self.base.transform(out)
            out = fm.frame
            families["base"] = list(self.base.FEATURE_NAMES)
            cols.extend(self.base.FEATURE_NAMES)
        if self.config.use_sessions:
            out = self.sessions.transform(out)
            families["sessions"] = list(self.sessions.FEATURE_NAMES)
            cols.extend(self.sessions.FEATURE_NAMES)
        if self.config.use_structure:
            out = self.structure.transform(out)
            families["structure"] = list(self.structure.FEATURE_NAMES)
            cols.extend(self.structure.FEATURE_NAMES)
        if self.config.use_intermarket:
            out = self.intermarket.transform(out, externals=externals, peer_metal=peer_metal)
            families["intermarket"] = list(self.intermarket.FEATURE_NAMES)
            cols.extend(self.intermarket.FEATURE_NAMES)
        if self.config.use_macro:
            out = self.macro.transform(out, events=events)
            families["macro"] = list(self.macro.FEATURE_NAMES)
            cols.extend(self.macro.FEATURE_NAMES)

        # Drop duplicates preserving order
        cols = list(dict.fromkeys(cols))
        FeatureRegistry.assert_no_label_leakage(cols)
        return BuiltFeatures(frame=out, feature_columns=cols, families=families)
