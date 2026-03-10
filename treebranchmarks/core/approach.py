"""
Approach: a single algorithm implementation for a given task.

An Approach wraps one concrete way to compute something (e.g. SHAP values
using a specific library/method).  It has two responsibilities:

1. run() — execute the algorithm and return timing + output.
2. estimate_runtime() — return a theoretical runtime estimate in seconds,
   or None if no complexity formula is available.

Runtime Estimation
------------------
estimate_runtime() encodes the algorithm's known complexity as a formula
over TreeParameters.  The formula has a machine-dependent scaling constant
that must be calibrated once per machine using calibrate().

Calibration
-----------
calibrate() runs the approach on a tiny fixed problem and fits the scaling
constant so that estimate_runtime() matches the measured time on that instance.
The constant is persisted in cache_root/calibration/{approach_name}.json.

If calibrate() has never been called, estimate_runtime() returns None.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
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

    elapsed_s: float        # wall-clock time for this invocation
    is_estimated: bool = False    # True if elapsed_s was extrapolated internally
    not_supported: bool = False   # True if the approach cannot handle this input
    memory_crash: bool = False    # True if the approach would exhaust memory on this input
    estimation_description: str = ""  # human-readable note on how estimation was done


# ---------------------------------------------------------------------------
# Approach ABC
# ---------------------------------------------------------------------------

class Approach(ABC):
    """
    Base class for a single algorithm implementation.

    Attributes
    ----------
    name : str
        Unique identifier shown in reports (e.g. "shap_tree_path_dependent").
    description : str
        Human-readable description shown in the HTML report.
    """

    name: str
    description: str = ""
    method: Optional[Method] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        """
        Execute the algorithm.

        Implementations should time themselves internally and return the
        elapsed wall-clock time in ApproachOutput.elapsed_s.

        X_background is None for approaches that do not use a background dataset.
        """

    # ------------------------------------------------------------------
    # Runtime estimation (optional — override if you know the complexity)
    # ------------------------------------------------------------------

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        """
        Return the *unscaled* complexity value for this parameter set.

        Example for O(T * L * n * m):
            return params.T * params.L * params.n * params.m

        The scaling constant (machine-dependent) is applied by estimate_runtime().
        Return None if no formula is known.
        """
        return None

    def estimate_runtime(
        self,
        params: TreeParameters,
        cache_root: Path = Path("cache"),
    ) -> Optional[float]:
        """
        Return estimated wall-clock runtime in seconds, or None.

        Requires calibrate() to have been called at least once on this machine.
        """
        unscaled = self.complexity_formula(params)
        if unscaled is None:
            return None

        constant = self._load_calibration_constant(cache_root)
        if constant is None:
            return None

        return constant * unscaled

    def calibrate(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        cache_root: Path = Path("cache"),
    ) -> float:
        """
        Fit the scaling constant by running this approach on the provided
        (small) instance and comparing measured time to the formula value.

        Returns the fitted constant and persists it to disk.
        """
        params = trained_model.params.with_run_params(
            n=len(X_explain), m=len(X_background)
        )
        unscaled = self.complexity_formula(params)
        if unscaled is None or unscaled == 0:
            raise ValueError(
                f"Approach '{self.name}' has no complexity formula; cannot calibrate."
            )

        output = self.run(trained_model, X_explain, X_background)
        constant = output.elapsed_s / unscaled

        self._save_calibration_constant(constant, cache_root)
        print(
            f"[approach:{self.name}] Calibrated constant = {constant:.2e} "
            f"(measured {output.elapsed_s:.3f}s on calibration instance)"
        )
        return constant

    # ------------------------------------------------------------------
    # Calibration persistence
    # ------------------------------------------------------------------

    def _calibration_path(self, cache_root: Path) -> Path:
        d = cache_root / "calibration"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.name}.json"

    def _save_calibration_constant(self, constant: float, cache_root: Path) -> None:
        path = self._calibration_path(cache_root)
        with open(path, "w") as f:
            json.dump({"constant": constant, "approach": self.name}, f, indent=2)

    def _load_calibration_constant(self, cache_root: Path) -> Optional[float]:
        path = self._calibration_path(cache_root)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)["constant"]
