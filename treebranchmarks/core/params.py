"""
Central parameter types for decision tree SHAP benchmarking.

The 6 complexity parameters that govern algorithm runtime:
  n  – number of rows in the data being explained
  m  – number of rows in the background dataset
  F  – number of features
  T  – number of trees in the ensemble
  D  – maximum tree depth
  L  – average number of leaves per tree

These are populated in two phases:
  1. After model training: T, D, L, F  (from the trained artifact)
  2. At task run time:    n, m          (from the data slices used for that run)

Use `with_run_params(n, m)` to produce a complete TreeParameters from a partially-filled one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class EnsembleType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"


@dataclass(frozen=True)
class TreeParameters:
    """
    Fully describes the structural complexity of a trained tree ensemble
    together with the sizes of the data involved in a specific benchmark run.

    Frozen so it can be used as a dict key and in hashes.
    """

    # Set at model-train time
    T: int              # number of trees
    D: int              # maximum depth across all trees
    L: float            # average number of leaves per tree
    F: int              # number of features
    ensemble_type: EnsembleType

    # Set at task run time (use with_run_params to fill these in)
    n: int = 0          # rows being explained
    m: int = 0          # rows in the background dataset (0 = not applicable)

    # ------------------------------------------------------------------
    # Derived / utility
    # ------------------------------------------------------------------

    def with_run_params(self, n: int, m: int = 0) -> TreeParameters:
        """Return a new TreeParameters with n and m filled in."""
        return TreeParameters(
            T=self.T, D=self.D, L=self.L, F=self.F,
            ensemble_type=self.ensemble_type,
            n=n, m=m,
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ensemble_type"] = self.ensemble_type.value
        return d

    def cache_key(self) -> str:
        """Stable, order-independent MD5 hash suitable for use as a cache key."""
        serialized = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def __str__(self) -> str:
        return (
            f"TreeParameters(n={self.n}, m={self.m}, F={self.F}, "
            f"T={self.T}, D={self.D}, L={self.L:.1f}, "
            f"ensemble={self.ensemble_type.value})"
        )


def partial_tree_params(
    T: int,
    D: int,
    L: float,
    F: int,
    ensemble_type: EnsembleType,
) -> TreeParameters:
    """
    Construct a TreeParameters with n=0 and m=0 (not yet known).
    Intended to be called right after model training; complete it later
    with `with_run_params(n, m)`.
    """
    return TreeParameters(T=T, D=D, L=L, F=F, ensemble_type=ensemble_type)
