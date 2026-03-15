"""
Woodelf algorithm implementations.

WoodelfHDApproach / WoodelfHDGPUApproach
    Calls woodelf_for_high_depth directly for all 4 task types.
    No depth limits, no built-in estimation.

OriginalWoodelfApproach / OriginalWoodelfGPUApproach
    Calls woodelf.simple_woodelf.calculate_path_dependent_metric /
    calculate_background_metric for all 4 task types.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import woodelf
from woodelf.cube_metric import ShapleyValues, ShapleyInteractionValues
from woodelf.high_depth_woodelf import woodelf_for_high_depth

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import WOODELF_HD, ORIGINAL_WOODELF


class WoodelfHDApproach(Approach):
    """Calls woodelf_for_high_depth directly. No depth limits or estimation."""

    name = "WoodelfHD"
    method = WOODELF_HD
    description = "woodelf_for_high_depth"
    GPU = False

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf_for_high_depth(trained_model.raw_model, X_explain, None, ShapleyValues(), GPU=self.GPU)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf_for_high_depth(trained_model.raw_model, X_explain, None, ShapleyInteractionValues(), GPU=self.GPU)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf_for_high_depth(trained_model.raw_model, X_explain, X_background, ShapleyValues(), GPU=self.GPU)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf_for_high_depth(trained_model.raw_model, X_explain, X_background, ShapleyInteractionValues(), GPU=self.GPU)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)


class WoodelfHDGPUApproach(WoodelfHDApproach):
    """WoodelfHDApproach with GPU=True (requires CuPy: pip install cupy)."""

    name = "WoodelfHD GPU"
    description = "woodelf_for_high_depth accelerated on GPU (CuPy required)."
    GPU = True


class OriginalWoodelfApproach(Approach):
    """Calls woodelf.simple_woodelf for all 4 task types."""

    name = "OriginalWoodelf"
    method = ORIGINAL_WOODELF
    description = "woodelf.simple_woodelf — calculate_path_dependent_metric / calculate_background_metric."
    GPU = False

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf.simple_woodelf.calculate_path_dependent_metric(
            trained_model.raw_model, X_explain, metric=ShapleyValues(), GPU=self.GPU,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf.simple_woodelf.calculate_path_dependent_metric(
            trained_model.raw_model, X_explain, metric=ShapleyInteractionValues(), GPU=self.GPU,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf.simple_woodelf.calculate_background_metric(
            trained_model.raw_model, X_explain, X_background, metric=ShapleyValues(), GPU=self.GPU,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf.simple_woodelf.calculate_background_metric(
            trained_model.raw_model, X_explain, X_background, metric=ShapleyInteractionValues(), GPU=self.GPU,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)


class OriginalWoodelfGPUApproach(OriginalWoodelfApproach):
    """OriginalWoodelfApproach with GPU=True (requires CuPy: pip install cupy)."""

    name = "OriginalWoodelf GPU"
    description = "woodelf.simple_woodelf accelerated on GPU (CuPy required)."
    GPU = True
