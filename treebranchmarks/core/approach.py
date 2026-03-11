"""
Approach: a single algorithm implementation that can handle one or more benchmark tasks.

An Approach implements any subset of the four task types:
  - path_dependent_shap
  - path_dependent_interactions
  - background_shap
  - background_shap_interactions

Tasks that the approach does not support return ApproachOutput(not_supported=True).

Each task method has a corresponding complexity_formula_* method that returns
the unscaled operation count, used for runtime estimation and calibration.
"""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from treebranchmarks.core.method import Method
from treebranchmarks.core.params import TreeParameters
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

    # ------------------------------------------------------------------
    # Complexity formulas (override those you support)
    # ------------------------------------------------------------------

    def complexity_formula_path_dependent_shap(self, params: TreeParameters) -> Optional[float]:
        return None

    def complexity_formula_path_dependent_interactions(self, params: TreeParameters) -> Optional[float]:
        return None

    def complexity_formula_background_shap(self, params: TreeParameters) -> Optional[float]:
        return None

    def complexity_formula_background_shap_interactions(self, params: TreeParameters) -> Optional[float]:
        return None

    # ------------------------------------------------------------------
    # Runtime estimation (optional — requires calibration)
    # ------------------------------------------------------------------

    def estimate_runtime(
        self,
        task_type: str,
        params: TreeParameters,
        cache_root: Path = Path("cache"),
    ) -> Optional[float]:
        """Return estimated wall-clock runtime in seconds, or None."""
        formula_fn = getattr(self, f"complexity_formula_{task_type}", None)
        if formula_fn is None:
            return None
        unscaled = formula_fn(params)
        if unscaled is None:
            return None
        constant = self._load_calibration_constant(task_type, cache_root)
        if constant is None:
            return None
        return constant * unscaled

    def calibrate(
        self,
        task_type: str,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        cache_root: Path = Path("cache"),
    ) -> float:
        """
        Fit the scaling constant by running this approach on the provided
        (small) instance and comparing measured time to the formula value.
        """
        params = trained_model.params.with_run_params(
            n=len(X_explain),
            m=len(X_background) if X_background is not None else 0,
        )
        formula_fn = getattr(self, f"complexity_formula_{task_type}", None)
        if formula_fn is None:
            raise ValueError(f"Approach '{self.name}' has no complexity formula for task '{task_type}'.")
        unscaled = formula_fn(params)
        if unscaled is None or unscaled == 0:
            raise ValueError(f"Approach '{self.name}' complexity formula returned {unscaled!r} for task '{task_type}'.")

        task_fn = getattr(self, task_type)
        output = task_fn(trained_model, X_explain, X_background)
        constant = output.elapsed_s / unscaled

        self._save_calibration_constant(task_type, constant, cache_root)
        print(
            f"[approach:{self.name}:{task_type}] Calibrated constant = {constant:.2e} "
            f"(measured {output.elapsed_s:.3f}s on calibration instance)"
        )
        return constant

    # ------------------------------------------------------------------
    # Calibration persistence
    # ------------------------------------------------------------------

    def _calibration_path(self, task_type: str, cache_root: Path) -> Path:
        d = cache_root / "calibration"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.name}_{task_type}.json"

    def _save_calibration_constant(self, task_type: str, constant: float, cache_root: Path) -> None:
        path = self._calibration_path(task_type, cache_root)
        with open(path, "w") as f:
            json.dump({"constant": constant, "approach": self.name, "task_type": task_type}, f, indent=2)

    def _load_calibration_constant(self, task_type: str, cache_root: Path) -> Optional[float]:
        path = self._calibration_path(task_type, cache_root)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)["constant"]
