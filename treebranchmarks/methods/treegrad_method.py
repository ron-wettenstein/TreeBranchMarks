"""
TreeGradApproach — path-dependent SHAP using the TreeGrad package.

TreeGrad repo: https://github.com/watml/TreeGrad
Install:       git clone https://github.com/watml/TreeGrad && pip install -e TreeGrad

Limitations
-----------
* Single-sample API: TreeGrad computes one row at a time, so this approach
  loops over all N rows of X_explain.  Expect it to be slow for large n.
* sklearn models only: TreeGrad reads sklearn's internal ``estimators_``
  attribute.  LightGBM and XGBoost models return not_supported.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
import sys
from pathlib import Path as _Path

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import TREEGRAD


class TreeGradApproach(Approach):
    """
    Path-dependent SHAP via TreeGrad (Beta-Shapley with α=1, β=1).

    Only ``path_dependent_shap`` is supported.
    Requires an sklearn tree ensemble with an ``estimators_`` attribute
    (RandomForest, GradientBoosting).  Returns not_supported for
    LightGBM / XGBoost models.
    """

    name = "TreeGrad"
    method = TREEGRAD
    description = (
        "Path-dependent SHAP via TreeGrad (single-sample loop, sklearn models only). "
        "See https://github.com/watml/TreeGrad"
    )

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        _treegrad_path = (_Path(__file__).parent / ".." / ".." / ".." / "third_party_repos" / "TreeGrad").resolve()
        if not _treegrad_path.exists():
            raise ImportError(
                f"TreeGrad directory not found at {_treegrad_path!r}. "
                "Clone it from https://github.com/watml/TreeGrad into that directory."
            )
        _treegrad_path_str = str(_treegrad_path)
        if _treegrad_path_str not in sys.path:
            sys.path.insert(0, _treegrad_path_str)
        try:
            from TreeGrad import treegrad_shap
        except ImportError as exc:
            raise ImportError(
                f"TreeGrad directory exists at {_treegrad_path!r} but could not be imported. "
                "Make sure the repo is not corrupted."
            ) from exc

        model = trained_model.raw_model

        if not hasattr(model, "estimators_"):
            return ApproachOutput(elapsed_s=0.0, not_supported=True)

        class_index = 0 if is_classifier(model) else None
        N = len(X_explain)
        MAX_ROWS = 200

        if N > MAX_ROWS:
            X_arr = X_explain.iloc[:MAX_ROWS].to_numpy()
            t0 = time.perf_counter()
            np.stack([
                treegrad_shap(model, row, semivalue=(1, 1), class_index=class_index)
                for row in X_arr
            ])
            elapsed = time.perf_counter() - t0
            return ApproachOutput(
                elapsed_s=elapsed * N / MAX_ROWS,
                is_estimated=True,
                estimation_description=f"ran on {MAX_ROWS} of {N} rows, extrapolated ×{N / MAX_ROWS:.1f}",
            )

        t0 = time.perf_counter()
        np.stack([
            treegrad_shap(model, row, semivalue=(1, 1), class_index=class_index)
            for row in X_explain.to_numpy()
        ])
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Quick sanity check — run as: python -m treebranchmarks.methods.treegrad_method
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification

    print("Building small dataset and model...")
    X_np, y_np = make_classification(n_samples=20000, n_features=30, random_state=0)
    X_df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(X_np.shape[1])])

    model = GradientBoostingClassifier(n_estimators=5, max_depth=24, random_state=0)
    model.fit(X_np, y_np)

    X_small = X_df.iloc[:100]

    # --- TreeGrad ---
    _treegrad_path = (_Path(__file__).parent / ".." / ".." / ".." / "third_party_repos" / "TreeGrad").resolve()
    if not _treegrad_path.exists():
        raise ImportError(f"TreeGrad not found at {_treegrad_path!r}")
    if str(_treegrad_path) not in sys.path:
        sys.path.insert(0, str(_treegrad_path))
    from TreeGrad import treegrad_shap

    print("Running TreeGrad...")
    t0 = time.perf_counter()
    treegrad_values = np.stack([
        treegrad_shap(model, row, semivalue=(1, 1), class_index=0)
        for row in X_small.to_numpy()
    ])
    treegrad_time = time.perf_counter() - t0
    print(f"  TreeGrad time : {treegrad_time:.4f}s  ({treegrad_time / len(X_small) * 1000:.2f} ms/sample)")

    treegrad_values = -1 * treegrad_values  # TreeGrad returns negative SHAP values; flip sign to match SHAP convention

    # --- SHAP ---
    print("Running SHAP...")
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_small)
    shap_time = time.perf_counter() - t0
    print(f"  SHAP time     : {shap_time:.4f}s  ({shap_time / len(X_small) * 1000:.2f} ms/sample)")
    # GradientBoostingClassifier returns a list [class0, class1]; take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # --- Compare ---
    mae = np.abs(treegrad_values - shap_values).mean()
    max_err = np.abs(treegrad_values - shap_values).max()

    print(f"\nResults over {len(X_small)} samples, {X_np.shape[1]} features:")
    print(f"  TreeGrad time : {treegrad_time:.4f}s")
    print(f"  SHAP time     : {shap_time:.4f}s")
    print(f"  TreeGrad mean : {treegrad_values.mean():.6f}")
    print(f"  SHAP mean     : {shap_values.mean():.6f}")
    print(f"  MAE           : {mae:.2e}")
    print(f"  Max abs error : {max_err:.2e}")
    print("\nTreeGrad values (first row):", np.round(treegrad_values[0], 6))
    print("SHAP values    (first row):", np.round(shap_values[0], 6))
    print("\nPASS" if mae < 1e-3 else f"\nWARNING: MAE {mae:.2e} exceeds 1e-3 — check alignment")
