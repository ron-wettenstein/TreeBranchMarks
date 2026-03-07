"""
Task: times all approaches for a given (model, X_explain, X_background) triple.

A Task owns a list of Approach objects.  When run(), it:
  1. Checks whether the projected runtime (extrapolated from prev_times) exceeds
     timeout_s.  If so, the approach is skipped and its time is estimated by
     linear scaling from the previous measurement.
  2. Warms up each approach that will actually run (one un-timed call)
  3. Runs each approach n_repeats times, recording wall-clock time
  4. Returns a TaskResult

TaskResult / ApproachResult are plain dataclasses that serialize cleanly to JSON.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from treebranchmarks.core.approach import Approach
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ApproachResult:
    approach_name: str
    running_time: float             # measured mean, or linear extrapolation if is_estimated
    std_time_s: float               # 0.0 when is_estimated=True
    is_estimated: bool              # True = run was skipped; time is linearly extrapolated
    error: Optional[str]            # filled if the approach raised an exception
    method: str = ""                # "shap" | "woodelf" | ""

    def as_dict(self) -> dict:
        return {
            "approach_name": self.approach_name,
            "running_time": self.running_time,
            "std_time_s": self.std_time_s,
            "is_estimated": self.is_estimated,
            "error": self.error,
            "method": self.method,
        }


@dataclass
class TaskResult:
    task_name: str
    params: TreeParameters
    approach_results: dict[str, ApproachResult]  # keyed by approach.name

    def as_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "params": self.params.as_dict(),
            "approach_results": {
                k: v.as_dict() for k, v in self.approach_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task:
    """
    Groups one or more Approaches under a shared task name and times them.

    Parameters
    ----------
    name : str
        Unique task identifier shown in reports.
    approaches : list[Approach]
        The algorithm implementations to benchmark.
    n_repeats : int
        How many timed repetitions per approach (default 3).
    warmup : bool
        Whether to run one un-timed warmup call before timing (default True).
    cache_root : Path
        Reserved for future calibration lookup.
    """

    def __init__(
        self,
        name: str,
        approaches: list[Approach],
        n_repeats: int = 3,
        warmup: bool = True,
        cache_root: Path = Path("cache"),
    ) -> None:
        self.name = name
        self.approaches = approaches
        self.n_repeats = n_repeats
        self.warmup = warmup
        self.cache_root = cache_root

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        prev_times: Optional[dict[str, tuple[int, float]]] = None,
        timeout_s: float = 600.0,
    ) -> TaskResult:
        """
        Time all approaches and return a TaskResult.

        Parameters
        ----------
        X_background : DataFrame or None
            None when no background dataset is needed (path-dependent tasks).
        prev_times : dict mapping approach_name → (prev_n, prev_measured_time_s)
            If provided, any approach whose linearly-extrapolated runtime to the
            current n exceeds timeout_s is skipped — its result is marked
            is_estimated=True with the extrapolated time as running_time.
        timeout_s : float
            Skip threshold in seconds (default 600 = 10 minutes).
        """
        params = trained_model.params.with_run_params(
            n=len(X_explain),
            m=len(X_background) if X_background is not None else 0,
        )
        current_n = len(X_explain)

        approach_results: dict[str, ApproachResult] = {}
        for approach in self.approaches:
            prev = (prev_times or {}).get(approach.name)
            result = self._time_approach(
                approach, trained_model, X_explain, X_background, params,
                prev_n_time=prev, current_n=current_n, timeout_s=timeout_s,
            )
            approach_results[approach.name] = result
            if result.error:
                print(f"  [approach:{approach.name}] ERROR: {result.error}")
            elif result.is_estimated:
                print(
                    f"  [approach:{approach.name}] "
                    f"ESTIMATED={result.running_time:.1f}s* (extrapolated, skipped)"
                )
            else:
                print(
                    f"  [approach:{approach.name}] "
                    f"mean={result.running_time:.3f}s ± {result.std_time_s:.3f}s"
                )

        return TaskResult(task_name=self.name, params=params, approach_results=approach_results)

    # ------------------------------------------------------------------
    # Internal timing logic
    # ------------------------------------------------------------------

    def _time_approach(
        self,
        approach: Approach,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        params: TreeParameters,
        prev_n_time: Optional[tuple[int, float]],
        current_n: int,
        timeout_s: float,
    ) -> ApproachResult:
        # Check whether extrapolated runtime exceeds the timeout.
        if prev_n_time is not None:
            prev_n, prev_time = prev_n_time
            if prev_n > 0:
                extrapolated = prev_time * (current_n / prev_n)
                if extrapolated > timeout_s:
                    return ApproachResult(
                        approach_name=approach.name,
                        running_time=extrapolated,
                        std_time_s=0.0,
                        is_estimated=True,
                        error=None,
                        method=getattr(approach, "method", ""),
                    )

        REPEAT_THRESHOLD_S = 10.0

        times: list[float] = []
        any_estimated = False
        error: Optional[str] = None

        try:
            if self.warmup:
                approach.run(trained_model, X_explain, X_background)

            # Always run once.
            first = approach.run(trained_model, X_explain, X_background)
            times.append(first.elapsed_s)
            if first.is_estimated:
                any_estimated = True

            # Only repeat if the first run was fast enough.
            if first.elapsed_s < REPEAT_THRESHOLD_S:
                for _ in range(self.n_repeats - 1):
                    output = approach.run(trained_model, X_explain, X_background)
                    times.append(output.elapsed_s)
                    if output.is_estimated:
                        any_estimated = True

        except Exception:
            error = traceback.format_exc()

        if times:
            mean_t = float(np.mean(times))
            std_t = float(np.std(times))
        else:
            mean_t = std_t = 0.0

        return ApproachResult(
            approach_name=approach.name,
            running_time=mean_t,
            std_time_s=std_t,
            is_estimated=any_estimated,
            error=error,
            method=getattr(approach, "method", ""),
        )
