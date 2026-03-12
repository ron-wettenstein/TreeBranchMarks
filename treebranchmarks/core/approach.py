"""
Approach: a single algorithm implementation that can handle one or more benchmark tasks.

An Approach implements any subset of the four task types:
  - path_dependent_shap
  - path_dependent_interactions
  - background_shap
  - background_shap_interactions

Tasks that the approach does not support return ApproachOutput(not_supported=True).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from treebranchmarks.core.method import Method
from treebranchmarks.core.model import TrainedModel


# ---------------------------------------------------------------------------
# ApproachOutput
# ---------------------------------------------------------------------------

@dataclass
class ApproachOutput:
    """Raw output of a single approach invocation."""

    elapsed_s: float          # wall-clock time for this invocation
    is_estimated: bool = False    # True if elapsed_s was extrapolated internally
    not_supported: bool = False   # True if the approach cannot handle this input
    memory_crash: bool = False    # True if the approach would exhaust memory on this input
    estimation_description: str = ""  # human-readable note on how estimation was done


# ---------------------------------------------------------------------------
# Approach ABC
# ---------------------------------------------------------------------------

class Approach(ABC):
    """
    Base class for a benchmarked algorithm implementation.

    Subclasses override any of the four task methods they support.
    Unsupported tasks automatically return not_supported=True.

    Attributes
    ----------
    name : str
        Unique identifier shown in reports.
    description : str
        Human-readable description shown in the HTML report.
    method : Method | None
        The Method this approach belongs to (drives scoring).
    """

    name: str
    description: str = ""
    method: Optional[Method] = None

    # ------------------------------------------------------------------
    # Task methods (override those you support)
    # ------------------------------------------------------------------

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return ApproachOutput(elapsed_s=0.0, not_supported=True)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return ApproachOutput(elapsed_s=0.0, not_supported=True)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return ApproachOutput(elapsed_s=0.0, not_supported=True)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return ApproachOutput(elapsed_s=0.0, not_supported=True)

