"""Tests for the treelite -> shapiq converter in methods.model_parsing.

The converter is the one piece of parsing logic this repo owns, so it is pinned against two
independent references: shapiq's own converter (structure) and the fitted model's own
``predict`` (semantics).
"""

import numpy as np
import pytest

from treebranchmarks.methods.model_parsing import (
    _FLOAT32_X_FAMILIES,
    _STRICT_LT_FAMILIES,
    _predict_shapiq_trees,
    _tree_structure_diff,
    ShapiqParser,
    TreeliteToShapiqParser,
    treelite_to_shapiq,
)

pytest.importorskip("treelite")
pytest.importorskip("shapiq")

SEED = 0
# Families shapiq can also parse, so the structural comparison has a reference to run against.
COMPARABLE = ["xgboost", "lightgbm", "random_forest", "extra_trees"]
# Families only reachable through treelite -> the coverage the bridge adds.
BRIDGE_ONLY = ["gradient_boosting", "hist_gradient_boosting"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(SEED)
    X = rng.random((600, 5))
    # A feature with few distinct values makes ties on thresholds likely, which is exactly
    # where float32/`<=` semantics bite.
    X[:, 4] = rng.integers(0, 6, size=600).astype(float)
    y_reg = X[:, 0] * 3.0 + X[:, 1] - 0.5 * X[:, 4]
    y_clf = (y_reg > np.median(y_reg)).astype(int)
    import pandas as pd
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y_reg, y_clf


def _fit(family, task, X, y, n_trees=12, max_depth=4):
    from treebranchmarks.methods.model_parsing import build_model
    return build_model(family, task, X, y, n_trees=n_trees, max_depth=max_depth, seed=SEED)


@pytest.mark.parametrize("family", COMPARABLE)
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_structure_matches_shapiq_native(family, task, data):
    """Our trees must be identical to shapiq's own, up to node renumbering."""
    X, y_reg, y_clf = data
    model = _fit(family, task, X, y_reg if task == "regression" else y_clf)
    ours = TreeliteToShapiqParser().parse(model, list(X.columns))
    native = ShapiqParser().parse(model, list(X.columns))

    assert len(ours) == len(native)
    diffs = [d for d in (_tree_structure_diff(a, b) for a, b in zip(ours, native)) if d]
    assert not diffs, f"{family}/{task}: {diffs[:3]}"


@pytest.mark.parametrize("family", COMPARABLE + BRIDGE_ONLY)
def test_predictions_match_the_model(family, data):
    """
    Traversing our trees must reproduce the model's own regression predictions, up to the
    constant intercept that treelite keeps outside the leaves.
    """
    X, y_reg, _ = data
    model = _fit(family, "regression", X, y_reg)
    trees = TreeliteToShapiqParser().parse(model, list(X.columns))

    probe = X.to_numpy(np.float64)
    pred = _predict_shapiq_trees(
        trees, probe,
        strict_lt=family in _STRICT_LT_FAMILIES,
        x_float32=family in _FLOAT32_X_FAMILIES,
    )
    truth = np.asarray(model.predict(X), dtype=np.float64).ravel()
    residual = truth - pred
    # float32 thresholds cap XGBoost's achievable agreement at ~1e-6.
    assert np.max(np.abs(residual - residual.mean())) < 1e-5


@pytest.mark.parametrize("family", BRIDGE_ONLY)
def test_bridge_covers_models_shapiq_rejects(family, data):
    """These are the families shapiq's native converter refuses; the bridge handles them."""
    X, y_reg, _ = data
    model = _fit(family, "regression", X, y_reg)

    with pytest.raises(Exception):
        ShapiqParser().parse(model, list(X.columns))

    trees = TreeliteToShapiqParser().parse(model, list(X.columns))
    assert len(trees) > 0


def test_leaf_and_internal_node_sentinels(data):
    """shapiq's conventions: leaves carry feature -2 and NaN thresholds."""
    X, y_reg, _ = data
    model = _fit("xgboost", "regression", X, y_reg)
    for tree in TreeliteToShapiqParser().parse(model, list(X.columns)):
        leaves = tree.children_left == -1
        assert np.all(tree.features[leaves] == -2)
        assert np.all(np.isnan(tree.thresholds[leaves]))
        assert np.all(tree.children_right[leaves] == -1)
        assert not np.any(np.isnan(tree.thresholds[~leaves]))
        assert np.all(tree.node_sample_weight >= 0)


def test_forest_leaf_values_are_averaged(data):
    """
    Forests set treelite's ``average_tree_output``, so leaf values must be divided by the
    tree count — otherwise the summed ensemble overshoots by a factor of n_trees.
    """
    X, y_reg, _ = data
    n_trees = 12
    model = _fit("random_forest", "regression", X, y_reg, n_trees=n_trees)
    trees = TreeliteToShapiqParser().parse(model, list(X.columns))

    summed = _predict_shapiq_trees(trees, X.to_numpy(np.float64), x_float32=True)
    truth = np.asarray(model.predict(X), dtype=np.float64)
    assert np.allclose(summed, truth, atol=1e-9)


def test_classifier_leaf_vectors_become_probabilities(data):
    """sklearn classifiers store per-leaf vote vectors; they must reduce to a probability."""
    X, _, y_clf = data
    model = _fit("random_forest", "classification", X, y_clf)
    trees = TreeliteToShapiqParser().parse(model, list(X.columns))

    proba = _predict_shapiq_trees(trees, X.to_numpy(np.float64), x_float32=True)
    assert np.all(proba >= -1e-9) and np.all(proba <= 1 + 1e-9)
    assert np.allclose(proba, model.predict_proba(X)[:, 1], atol=1e-9)


def test_split_less_trees_are_converted(data):
    """
    Boosters emit bare-root-leaf trees once the loss stops improving — 58 of 100 trees on an
    unlimited-depth California Housing model. shapiq's TreeModel derives ``max_feature_id``
    with ``max()`` over a then-empty set, so these must be handled explicitly.
    """
    import xgboost as xgb
    X, y_reg, _ = data
    # min_child_weight above the total hessian makes every split impossible -> all stumps.
    model = xgb.XGBRegressor(n_estimators=5, max_depth=0, min_child_weight=10_000,
                             tree_method="hist", random_state=SEED, verbosity=0).fit(X, y_reg)

    trees = TreeliteToShapiqParser().parse(model, list(X.columns))
    assert len(trees) == 5
    for tree in trees:
        assert len(tree.children_left) == 1          # root only
        assert bool(tree.leaf_mask[0])
        assert tree.n_features_in_tree == 0

    pred = _predict_shapiq_trees(trees, X.to_numpy(np.float64), strict_lt=True, x_float32=True)
    truth = np.asarray(model.predict(X), dtype=np.float64)
    residual = truth - pred
    assert np.max(np.abs(residual - residual.mean())) < 1e-5


def test_treelite_to_shapiq_accepts_a_parsed_treelite_model(data):
    """The converter works on any treelite model, not just ones built by our parser."""
    import treelite
    X, y_reg, _ = data
    model = _fit("lightgbm", "regression", X, y_reg)
    tl_model = treelite.frontend.from_lightgbm(model.booster_)

    trees = treelite_to_shapiq(tl_model)
    assert len(trees) == tl_model.num_tree
