"""
Subprocess runner for WoodelfVersionApproach.

Called by WoodelfVersionApproach._run() as a subprocess, with a different
woodelf installation on PYTHONPATH.  Reads a JSON task spec from stdin,
runs WoodelfExplainer, and writes a JSON result to stdout.

The timing wraps only the shap_values / shap_interaction_values call so
subprocess startup and data loading are excluded from the measurement.
"""

import json
import os
import pickle
import sys
import time

import pandas as pd
from woodelf import WoodelfExplainer


def main() -> None:
    args = json.loads(sys.stdin.read())

    with open(args["model_pickle_path"], "rb") as f:
        raw_model = pickle.load(f)

    X_explain: pd.DataFrame = pd.read_pickle(args["x_explain_path"])

    x_bg_path = args.get("x_background_path")
    X_background: pd.DataFrame | None = pd.read_pickle(x_bg_path) if x_bg_path else None

    D: int = args["D"]
    T: int = args["T"]
    tree_limit_depth: int | None = args.get("tree_limit_depth")
    feature_perturbation: str = args["feature_perturbation"]
    use_interactions: bool = args["use_interactions"]
    gpu: bool = args.get("gpu", False)

    if feature_perturbation == "interventional":
        explainer = WoodelfExplainer(
            raw_model,
            data=X_background,
            feature_perturbation="interventional",
            GPU=gpu,
        )
    else:
        explainer = WoodelfExplainer(
            raw_model,
            feature_perturbation="tree_path_dependent",
            GPU=gpu,
        )

    if tree_limit_depth is not None and D > tree_limit_depth:
        t0 = time.perf_counter()
        if use_interactions:
            explainer.shap_interaction_values(
                X_explain, tree_limit=1,
                include_interaction_with_itself=False,
                as_df=True, exclude_zero_contribution_features=True,
            )
        else:
            explainer.shap_values(X_explain, tree_limit=1)
        elapsed = time.perf_counter() - t0
        result = {
            "elapsed_s": elapsed * T,
            "is_estimated": True,
            "estimation_description": (
                f"D={D} > {tree_limit_depth}: "
                f"ran with tree_limit=1, extrapolated ×{T} trees"
            ),
        }
    else:
        t0 = time.perf_counter()
        if use_interactions:
            explainer.shap_interaction_values(
                X_explain,
                include_interaction_with_itself=False,
                as_df=True, exclude_zero_contribution_features=True,
            )
        else:
            explainer.shap_values(X_explain)
        elapsed = time.perf_counter() - t0
        result = {
            "elapsed_s": elapsed,
            "is_estimated": False,
            "estimation_description": "",
        }

    os.write(1, ("TREEBRANCHMARKS_RESULT:" + json.dumps(result) + "\n").encode())


if __name__ == "__main__":
    main()
