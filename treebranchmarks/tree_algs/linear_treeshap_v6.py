"""
Linear TreeSHAP V6 Vectorized — woodelf API.

Same algorithm as v6_vectorized.py (telescoping + quadrature, no pending lists),
adapted to use woodelf's DecisionTreeNode / DecisionTreesEnsemble API instead
of sklearn's internal tree_ arrays.

Key differences vs v6_vectorized.py
-------------------------------------
  - Tree traversal : woodelf DecisionTreeNode objects (node.left, node.right,
                     node.feature_name, node.value, node.cover, node.is_leaf()).
  - Input X        : pandas DataFrame — column access by feature name.
  - Ensemble       : summed over all trees in DecisionTreesEnsemble.trees.
  - edge weight    : w_e = child.cover / node.cover
  - leaf value     : node.value  (when node.is_leaf())
  - satisfies      : node.shall_go_left(X)  →  numpy bool array

Shape conventions
-----------------
  N         number of test samples (len(X))
  F         number of features (len(X.columns))
  n_quad    number of quadrature points (N_QUAD = 12)

  c_vals    (N, n_quad)
  p_vals    (N, F)         per-sample accumulated-q; _UNSEEN = not yet seen
  phi       (N, F)         SHAP values (summed over all trees for ensembles)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from woodelf.decision_trees_ensemble import DecisionTreeNode, DecisionTreesEnsemble
from woodelf.parse_models import load_decision_tree_ensemble_model  # noqa: F401

DTYPE = np.longdouble
Q_EPS = 1e-15

# Gauss-Legendre quadrature nodes and weights on [0, 1]
N_QUAD = 16
_nodes_11, _weights_11 = np.polynomial.legendre.leggauss(N_QUAD)
QUAD_NODES = DTYPE(0.5) * (_nodes_11.astype(DTYPE) + DTYPE(1.0))   # (n_quad,)
QUAD_WEIGHTS = DTYPE(0.5) * _weights_11.astype(DTYPE)               # (n_quad,)

# Sentinel stored in p_vals for features not yet seen on the current path
_UNSEEN = DTYPE(-999.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_terms_vec(h_vals: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
    """
    Compute  (p − 1) · ∫₀¹ H(t) / (1 + (p−1)·t) dt  for every sample.

    Parameters
    ----------
    h_vals : (N, n_quad)
    p_vec  : (N,)  — _UNSEEN or near-1 → contribution 0

    Returns
    -------
    (N,)
    """
    alpha  = p_vec - DTYPE(1.0)
    silent = (p_vec == _UNSEEN) | (np.abs(alpha) < Q_EPS)
    denom  = DTYPE(1.0) + alpha[:, None] * QUAD_NODES[None, :]
    result = alpha * np.dot(h_vals / denom, QUAD_WEIGHTS)
    return (~silent) * result


def _expected_value(root: DecisionTreeNode) -> float:
    """E[f(X_train)] = training-sample-weighted average of all leaf values."""
    leaves     = root.get_all_leaves()
    total_cover = sum(leaf.cover for leaf in leaves)
    return float(sum(leaf.value * leaf.cover for leaf in leaves) / total_cover)


def _shall_go_left(node: DecisionTreeNode, X: pd.DataFrame) -> np.ndarray:
    """
    Return a (N,) bool numpy array: True for samples that go LEFT at node.
    Wraps node.shall_go_left so callers always get a plain ndarray.
    """
    result = node.shall_go_left(X)
    return np.asarray(result, dtype=bool)


# ---------------------------------------------------------------------------
# Per-tree SHAP DFS
# ---------------------------------------------------------------------------

def _shap_single_tree(
    root: DecisionTreeNode,
    X: pd.DataFrame,
    n_features: int,
    feature_to_idx: dict[str, int],
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for one tree root over all rows of X.

    Parameters
    ----------
    root           : woodelf tree root node
    X              : (N, F) DataFrame
    n_features     : total number of features (= len(X.columns))
    feature_to_idx : {feature_name: column_index}

    Returns
    -------
    phi      : (N, n_features) SHAP values
    expected : scalar baseline E[f(X_train)]
    """
    N = len(X)
    phi   = np.zeros((N, n_features), dtype=DTYPE)
    p_vals = np.full((N, n_features), _UNSEEN, dtype=DTYPE)

    def dfs(node: DecisionTreeNode, c_vals: np.ndarray, w_prod: float) -> np.ndarray:
        """
        Parameters
        ----------
        node   : current tree node
        c_vals : (N, n_quad)
        w_prod : float  (cumulative edge-weight product — scalar, tree property)

        Returns
        -------
        H_node : (N, n_quad)
        """
        if node.is_leaf():
            return c_vals * DTYPE(node.value * w_prod)

        f_idx = feature_to_idx[node.feature_name]
        child_h = [None, None]

        goes_left = _shall_go_left(node, X)           # (N,) bool, computed once

        for idx, (child, is_left) in enumerate(
            [(node.left, True), (node.right, False)]
        ):
            w_e: float = child.cover / node.cover
            satisfies  = goes_left if is_left else ~goes_left  # (N,) bool

            # Get the old shapley values of the current node's feature
            p_old = p_vals[:, f_idx].copy()
            is_unseen = (p_old == _UNSEEN)
            is_normal = ~is_unseen & (np.abs(p_old) >= Q_EPS)

            # (1 if we encounter this feature for the first time, else its old shapley value or 0 if old shapley value numerically tinny, less than 10e-15) 
            p_up       = is_unseen + is_normal * p_old
            # (1 if we encounter this feature for the first time, else its old shapley value) 
            safe_p_old = is_unseen + (~is_unseen) * p_old
            p_e_pre    = (is_unseen | is_normal) * (safe_p_old / w_e)
            p_e        = satisfies * p_e_pre

            alpha_e = p_e - DTYPE(1.0)
            c_child = c_vals * (
                DTYPE(1.0) + alpha_e[:, None] * QUAD_NODES[None, :]
            )

            safe_alpha_old = (~is_unseen) * (p_old - DTYPE(1.0))
            should_divide  = (~is_unseen) & (np.abs(safe_alpha_old) >= Q_EPS)

            if np.any(should_divide):
                denom_old = (
                    DTYPE(1.0) + safe_alpha_old[:, None] * QUAD_NODES[None, :]
                )
                c_divided = c_child / denom_old
                c_child = (
                    should_divide[:, None] * c_divided
                    + (~should_divide)[:, None] * c_child
                )

            p_vals[:, f_idx] = p_e
            h_child = dfs(child, c_child, w_prod * w_e)
            p_vals[:, f_idx] = p_old

            phi[:, f_idx] += _extract_terms_vec(h_child, p_e)
            phi[:, f_idx] -= _extract_terms_vec(h_child, p_up)

            child_h[idx] = h_child

        return child_h[0] + child_h[1]

    c_init = np.ones((N, N_QUAD), dtype=DTYPE)
    dfs(root, c_init, 1.0)
    return phi, _expected_value(root)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def linear_treeshap_v6_woodelf(
    ensemble: DecisionTreesEnsemble,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for every row of X using the woodelf tree API.

    Supports both single-tree and ensemble models (random forests, gradient
    boosting).  For ensembles, phi and expected are summed over all trees.

    Parameters
    ----------
    ensemble : DecisionTreesEnsemble
        Loaded via woodelf.parse_models.load_decision_tree_ensemble_model.
    X        : (N, F) pandas DataFrame

    Returns
    -------
    phi      : (N, F) numpy array of SHAP values
    expected : float  E[f(X_train)] baseline (summed over trees)
    """
    feature_names  = list(X.columns)
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    n_features     = len(feature_names)
    N              = len(X)

    phi_total      = np.zeros((N, n_features), dtype=DTYPE)
    expected_total = 0.0

    for root in ensemble.trees:
        phi, expected = _shap_single_tree(root, X, n_features, feature_to_idx)
        phi_total      += phi
        expected_total += expected

    return phi_total, expected_total
