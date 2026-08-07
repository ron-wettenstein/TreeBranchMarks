"""
Model-parsing benchmark: how long does it take to turn a trained tree ensemble
into an explainer-ready in-memory representation?

Every TreeSHAP-style explainer starts by *parsing* the model — walking the native
booster/forest and materialising a library-specific tree structure. That step is pure
overhead: it happens before a single Shapley value is computed, and it is paid again on
every explainer construction. This module measures it.

Parsers compared
----------------
1. ``TreeliteParser``          — ``treelite.frontend.from_xgboost`` / ``from_lightgbm`` /
                                 ``treelite.sklearn.import_model``. A compiler front-end,
                                 not an explainer, included as a fast baseline.
2. ``ShapParser``              — ``shap.explainers._tree.TreeEnsemble(model)``, the parsing
                                 half of ``shap.TreeExplainer``.
3. ``ShapiqParser``            — ``shapiq.explainer.tree.validation.validate_tree_model``,
                                 which dispatches to ``shapiq/explainer/tree/conversion/*``.
4. ``WoodelfShapParser``       — ``woodelf...parse_models_using_shap.load_model_using_shap``.
5. ``WoodelfTreeliteParser``   — ``woodelf...parse_models_using_treelite.load_model_using_treelite``.

   These two deliberately call woodelf's *backend-specific* loaders rather than its
   ``parse_models.load_decision_tree_ensemble_model`` dispatcher. The dispatcher picks an
   engine per model class, which is the right default everywhere else in the repo but would
   defeat the purpose here: the point is to measure shap-backed against treelite-backed
   parsing on the same model. Both return the same ``DecisionTreesEnsemble``, so the
   difference is the upstream front-end, not the woodelf node objects built on top.

6. ``TreeliteToShapiqParser``  — **written here**: parse with treelite, then convert
                                 treelite's arrays into shapiq's ``TreeModel`` format.
                                 See :func:`treelite_to_shapiq`.

Why parser 5 is interesting
---------------------------
It is not just a slower route to the same place. It is both faster than shapiq's own
converter and able to parse models shapiq rejects:

* **Speed.** shapiq's booster converters go through ``trees_to_dataframe()`` and a chain of
  ``pandas.replace`` calls. Measured on a 1000-tree XGBoost model: shapiq 13.5 s vs. 0.40 s
  through treelite — ~34x. The gap widens with depth (52 s vs. 0.25 s at depth 12).
* **Model coverage.** shapiq's converter accepts a fixed list of sklearn/xgboost/lightgbm
  classes and rejects ``GradientBoosting*`` and ``HistGradientBoosting*`` outright.
* **Split-less trees.** Boosters emit bare-root-leaf trees once the loss stops improving
  (58 of 100 trees on an unlimited-depth California Housing model). shapiq's ``TreeModel``
  computes ``max_feature_id`` with ``max()`` over the tree's non-leaf features and raises
  ``ValueError`` on the empty set, so shapiq cannot parse such a model at all — through
  either its own converter or a naive bridge. :func:`treelite_to_shapiq` handles them.

The one place it loses is scikit-learn forests, whose arrays shapiq and shap can already
consume almost directly; paying for a treelite import there is a 2-6x slowdown.

Experiments
-----------
``model_types``  — every parser against a zoo of model families (XGBoost, LightGBM,
                   RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting,
                   DecisionTree; classifier and regressor). Produces a support matrix plus
                   timings, and records *why* each unsupported combination fails.
``tree_count``   — parse time vs ensemble size at 10 / 50 / 100 / 500 / 1000 trees.
``depth``        — parse time vs ``max_depth``, which drives node count super-linearly.

Run
---
    python -m treebranchmarks.methods.model_parsing            # all three experiments
    python -m treebranchmarks.methods.model_parsing --quick    # small, for a smoke test
    python -m treebranchmarks.methods.model_parsing --experiment tree_count
"""

from __future__ import annotations

import argparse
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# treelite -> shapiq conversion (parser 5)
# ---------------------------------------------------------------------------

def treelite_to_shapiq(tl_model: Any, class_label: int = 1) -> list:
    """
    Convert a parsed treelite model into shapiq's ``list[TreeModel]`` format.

    treelite stores each tree as a struct-of-arrays, which is almost exactly shapiq's
    layout, so the conversion is a handful of vectorised operations per tree rather than a
    recursive walk. Fields are read through ``get_tree_accessor().get_field()`` (direct
    array access); the alternative ``dump_as_json()`` route materialises the whole ensemble
    as a JSON string first and is dramatically slower on large models.

    Layout differences that this function has to reconcile:

    * **Leaf sentinels.** treelite marks leaves with ``cleft == -1`` and ``split_index == -1``;
      shapiq expects ``features == -2`` at leaves and ``thresholds == NaN``.
    * **Scalar vs. vector leaves.** sklearn *classifiers* store a per-class vote vector per
      leaf (``leaf_vector`` + ``leaf_vector_begin/end``), while boosters store a scalar
      ``leaf_value``. Vector leaves are normalised to class probabilities and reduced to
      ``class_label``, matching what shapiq's own sklearn converter does.
    * **Averaged output.** Forests set ``average_tree_output``; their leaf values must be
      scaled by ``1 / num_tree`` so the ensemble sums rather than averages, again matching
      shapiq.
    * **Node weights.** treelite exposes ``sum_hess`` for boosters and ``data_count`` for
      LightGBM/sklearn, and never both — whichever is populated is used.

    Note on threshold semantics: treelite records a per-node ``comparison_op`` (``<`` for
    XGBoost, ``<=`` for LightGBM/sklearn), but shapiq's ``TreeModel`` has no slot for it and
    its traversal assumes ``<=``. This function therefore inherits shapiq's convention, which
    differs from XGBoost's only for feature values exactly equal to a threshold.

    Parameters
    ----------
    tl_model : treelite.Model
        An already-parsed treelite model.
    class_label : int
        Which class to extract from vector-valued leaves. Ignored for scalar leaves.

    Returns
    -------
    list[shapiq.explainer.tree.base.TreeModel]
    """
    from shapiq.explainer.tree.base import TreeModel

    header = tl_model.get_header_accessor()
    leaf_shape = _field(header, "leaf_vector_shape", dtype=np.int64)
    vector_len = int(leaf_shape[-1]) if leaf_shape is not None and leaf_shape.size else 1
    avg = _field(header, "average_tree_output", dtype=np.int64)
    average_output = bool(avg[0]) if avg is not None and avg.size else False

    n_trees = tl_model.num_tree
    scaling = 1.0 / n_trees if average_output else 1.0

    trees = []
    for i in range(n_trees):
        acc = tl_model.get_tree_accessor(i)
        children_left = _field(acc, "cleft", dtype=np.int64)
        children_right = _field(acc, "cright", dtype=np.int64)
        split_index = _field(acc, "split_index", dtype=np.int64)
        thresholds = _field(acc, "threshold", dtype=np.float64)

        n_nodes = children_left.size
        leaf_mask = children_left == -1

        features = split_index.copy()
        features[leaf_mask] = -2                      # shapiq's leaf sentinel

        thresholds = thresholds.astype(np.float64, copy=True)
        thresholds[leaf_mask] = np.nan

        values = _leaf_values(acc, leaf_mask, n_nodes, vector_len, class_label) * scaling

        # A split-less tree (a bare root leaf) contributes a constant. Boosters emit these
        # once the loss stops improving, which happens routinely at unlimited depth. shapiq's
        # __post_init__ derives max_feature_id via max() over the non-leaf features and
        # raises ValueError on the empty set, so those fields are supplied explicitly here.
        degenerate = not (~leaf_mask).any()
        extra_fields: dict[str, Any] = (
            {"n_features_in_tree": 0, "max_feature_id": 0, "feature_ids": set()}
            if degenerate else {}
        )

        trees.append(
            TreeModel(
                children_left=children_left,
                children_right=children_right,
                features=features,
                thresholds=thresholds,
                values=values,
                node_sample_weight=_node_weights(acc, n_nodes),
                empty_prediction=None,                # shapiq computes it lazily
                **extra_fields,
            )
        )
    return trees


def _field(accessor: Any, name: str, dtype: Any = None) -> Optional[np.ndarray]:
    """Read a treelite field, returning None when the model does not populate it."""
    try:
        arr = np.asarray(accessor.get_field(name)).ravel()
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _leaf_values(
    accessor: Any, leaf_mask: np.ndarray, n_nodes: int, vector_len: int, class_label: int
) -> np.ndarray:
    """Leaf values as a dense per-node array (internal nodes are 0.0)."""
    if vector_len > 1:
        # sklearn classifiers: per-leaf class vote vector, sliced out of one flat array.
        flat = _field(accessor, "leaf_vector", dtype=np.float64)
        begin = _field(accessor, "leaf_vector_begin", dtype=np.int64)
        end = _field(accessor, "leaf_vector_end", dtype=np.int64)
        values = np.zeros(n_nodes, dtype=np.float64)
        if flat is None or begin is None or end is None:
            return values
        idx = class_label if class_label < vector_len else vector_len - 1
        for node in np.flatnonzero(leaf_mask):
            vec = flat[begin[node]:end[node]]
            total = vec.sum()
            if total > 0 and vec.size > idx:
                values[node] = vec[idx] / total     # votes -> probability, as shapiq does
        return values

    leaf_value = _field(accessor, "leaf_value", dtype=np.float64)
    if leaf_value is None:
        return np.zeros(n_nodes, dtype=np.float64)
    return np.where(leaf_mask, leaf_value, 0.0)


def _node_weights(accessor: Any, n_nodes: int) -> np.ndarray:
    """Node sample weights: sum_hess for boosters, data_count elsewhere."""
    for name in ("sum_hess", "data_count"):
        arr = _field(accessor, name, dtype=np.float64)
        if arr is not None and arr.size == n_nodes:
            return arr
    return np.ones(n_nodes, dtype=np.float64)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

@dataclass
class ParsedStats:
    """Normalised view of a parsed model, so parsers can be compared like-for-like."""
    n_trees: Optional[int] = None
    n_nodes: Optional[int] = None


class ModelParser(ABC):
    """One way of turning a fitted model into an in-memory tree representation."""

    name: str
    label: str
    description: str = ""

    @abstractmethod
    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        """Parse the model. Must do the real work — no lazy handles."""

    def stats(self, parsed: Any) -> ParsedStats:      # best-effort, for sanity checks
        return ParsedStats()


class TreeliteParser(ModelParser):
    name = "treelite"
    label = "treelite"
    description = "treelite front-end (compiler IR, not an explainer format)."

    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        import treelite
        import lightgbm as lgb
        import xgboost as xgb

        if isinstance(model, xgb.XGBModel):
            return treelite.frontend.from_xgboost(model.get_booster())
        if isinstance(model, lgb.LGBMModel):
            return treelite.frontend.from_lightgbm(model.booster_)
        return treelite.sklearn.import_model(model)

    def stats(self, parsed: Any) -> ParsedStats:
        n = parsed.num_tree
        nodes = sum(
            int(np.asarray(parsed.get_tree_accessor(i).get_field("cleft")).size)
            for i in range(n)
        )
        return ParsedStats(n_trees=n, n_nodes=nodes)


class ShapParser(ModelParser):
    name = "shap"
    label = "shap (TreeEnsemble)"
    description = "shap.explainers._tree.TreeEnsemble — the parsing half of TreeExplainer."

    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        from shap.explainers._tree import TreeEnsemble
        return TreeEnsemble(model)

    def stats(self, parsed: Any) -> ParsedStats:
        return ParsedStats(
            n_trees=len(parsed.trees),
            n_nodes=int(sum(len(t.children_left) for t in parsed.trees)),
        )


class ShapiqParser(ModelParser):
    name = "shapiq"
    label = "shapiq"
    description = "shapiq.explainer.tree.validation.validate_tree_model."

    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        from shapiq.explainer.tree.validation import validate_tree_model
        return validate_tree_model(model)

    def stats(self, parsed: Any) -> ParsedStats:
        trees = parsed if isinstance(parsed, list) else [parsed]
        return ParsedStats(
            n_trees=len(trees),
            n_nodes=int(sum(len(t.children_left) for t in trees)),
        )


class _WoodelfParserBase(ModelParser):
    """
    Shared node counting for woodelf's two front-ends.

    woodelf exposes the same ``load_decision_tree_ensemble_model(model, features)`` entry
    point from two modules, backed by different upstream parsers. Both produce the same
    ``DecisionTreesEnsemble``, so timing them separately isolates the cost of the front-end
    (shap vs. treelite) from the cost of building woodelf's own node objects.
    """

    def stats(self, parsed: Any) -> ParsedStats:
        # woodelf builds linked node objects rather than arrays; count via BFS.
        n_nodes = 0
        for tree in parsed.trees:
            try:
                n_nodes += sum(1 for _ in tree.bfs())
            except Exception:
                return ParsedStats(n_trees=len(parsed.trees))
        return ParsedStats(n_trees=len(parsed.trees), n_nodes=n_nodes)


class WoodelfShapParser(_WoodelfParserBase):
    name = "woodelf_shap"
    label = "woodelf (via shap)"
    description = (
        "woodelf.core.trees.parse_models_using_shap.load_model_using_shap — "
        "woodelf's ensemble built on top of shap's TreeEnsemble parse."
    )

    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        from woodelf.core.trees.parse_models_using_shap import load_model_using_shap
        return load_model_using_shap(model, list(feature_names))


class WoodelfTreeliteParser(_WoodelfParserBase):
    name = "woodelf_treelite"
    label = "woodelf (via treelite)"
    description = (
        "woodelf.core.trees.parse_models_using_treelite.load_model_using_treelite — "
        "the same ensemble built on top of a treelite parse."
    )

    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        from woodelf.core.trees.parse_models_using_treelite import load_model_using_treelite
        return load_model_using_treelite(model, list(feature_names))


class TreeliteToShapiqParser(ModelParser):
    name = "treelite_shapiq"
    label = "treelite -> shapiq (ours)"
    description = (
        "Parse with treelite, then convert treelite's arrays into shapiq TreeModel objects. "
        "Timed end-to-end, so it includes the treelite parse it builds on."
    )

    def parse(self, model: Any, feature_names: Sequence[str]) -> Any:
        tl_model = TreeliteParser().parse(model, feature_names)
        return treelite_to_shapiq(tl_model)

    def stats(self, parsed: Any) -> ParsedStats:
        return ParsedStats(
            n_trees=len(parsed),
            n_nodes=int(sum(len(t.children_left) for t in parsed)),
        )


ALL_PARSERS: list[ModelParser] = [
    TreeliteParser(), ShapParser(), ShapiqParser(),
    WoodelfShapParser(), WoodelfTreeliteParser(), TreeliteToShapiqParser(),
]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    parser: str
    model_key: str
    dataset: str
    ok: bool
    median_s: float = float("nan")
    min_s: float = float("nan")
    n_trees: Optional[int] = None
    n_nodes: Optional[int] = None
    error_type: str = ""
    error_msg: str = ""
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {k: v for k, v in self.__dict__.items() if k != "extra"}
        row.update(self.extra)
        return row


def time_parse(
    parser: ModelParser,
    model: Any,
    feature_names: Sequence[str],
    dataset: str,
    model_key: str,
    repeats: int = 5,
    warmup: int = 1,
    budget_s: float = 20.0,
    **extra: Any,
) -> ParseResult:
    """
    Time one parser on one model.

    A warmup pass absorbs first-call import and JIT costs so they are not attributed to the
    parser. Both median and min are reported: min is the cleanest signal for a CPU-bound
    routine, median is the more honest summary when the machine is noisy.

    ``budget_s`` bounds the cost of a single configuration. The warmup pass is timed, and if
    it alone exceeds the budget the repeats are skipped and that one measurement is reported
    with ``repeats_run=1``. Without this the sweep is unbounded: shapiq needs ~52 s to parse
    one depth-12 XGBoost model and grows roughly 3.5x per two levels of depth, so six passes
    at unlimited depth would run for hours.
    """
    try:
        t0 = time.perf_counter()
        parsed = parser.parse(model, feature_names)      # warmup, and proves support
        first_s = time.perf_counter() - t0
    except Exception as exc:
        return ParseResult(
            parser=parser.name, model_key=model_key, dataset=dataset, ok=False,
            error_type=type(exc).__name__,
            error_msg=(str(exc).splitlines() or [""])[0][:200],
            extra=extra,
        )

    stats = ParsedStats()
    try:
        stats = parser.stats(parsed)
    except Exception:
        pass

    if first_s > budget_s:
        # One (cold) sample is all this configuration gets.
        return ParseResult(
            parser=parser.name, model_key=model_key, dataset=dataset, ok=True,
            median_s=first_s, min_s=first_s,
            n_trees=stats.n_trees, n_nodes=stats.n_nodes,
            extra={**extra, "repeats_run": 1, "over_budget": True},
        )

    for _ in range(max(0, warmup - 1)):
        parser.parse(model, feature_names)

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        parser.parse(model, feature_names)
        timings.append(time.perf_counter() - t0)
        if sum(timings) > budget_s:
            break                                        # stop early on slow parsers

    return ParseResult(
        parser=parser.name, model_key=model_key, dataset=dataset, ok=True,
        median_s=float(np.median(timings)), min_s=float(np.min(timings)),
        n_trees=stats.n_trees, n_nodes=stats.n_nodes,
        extra={**extra, "repeats_run": len(timings), "over_budget": False},
    )


def _skipped(parser: str, model_key: str, dataset: str, threshold_s: float, **extra: Any) -> ParseResult:
    """Placeholder row for a (parser, family) pair retired for being too slow."""
    return ParseResult(
        parser=parser, model_key=model_key, dataset=dataset, ok=False,
        error_type="SkippedTooSlow",
        error_msg=f"exceeded {threshold_s}s at a smaller setting; larger ones not attempted",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Datasets and model zoo
# ---------------------------------------------------------------------------

def find_repo_root(start: Optional[Path] = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return here


REPO_ROOT = find_repo_root(Path(__file__).parent)
CACHE_ROOT = REPO_ROOT / "cache"
OUT_DIR = REPO_ROOT / "results" / "model_parsing"


def load_dataset(name: str, n_rows: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, str]:
    """
    Load a small slice of a benchmark dataset.

    Parse time depends on the *tree structure*, not the row count, so only enough rows to
    grow realistic trees are needed. Feature count does matter (it shapes the trees), which
    is why the two defaults differ in width: 8 features vs 28.
    """
    if name == "california_housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = np.asarray(data.target)
        task = "regression"
    else:
        cache_dir = CACHE_ROOT / "datasets" / name
        if not (cache_dir / "X.parquet").exists():
            raise FileNotFoundError(
                f"No cached dataset at {cache_dir}. Run a treebranchmarks experiment that "
                f"loads '{name}' first, or pick --datasets california_housing."
            )
        X = pd.read_parquet(cache_dir / "X.parquet")
        y = np.load(cache_dir / "y.npy")
        task = "classification" if len(np.unique(y[:10000])) <= 20 else "regression"

    if len(X) > n_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    return X, y, task


def build_model(
    family: str, task: str, X: pd.DataFrame, y: np.ndarray,
    n_trees: int = 100, max_depth: Optional[int] = 6, seed: int = 0,
) -> Any:
    """Fit one model of the requested family. ``max_depth=None`` means unlimited."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import (
        ExtraTreesClassifier, ExtraTreesRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    clf = task == "classification"

    if family == "xgboost":
        cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
        # XGBoost spells "no limit" as 0, not None.
        return cls(n_estimators=n_trees, max_depth=0 if max_depth is None else max_depth,
                   tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0).fit(X, y)
    if family == "lightgbm":
        cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
        return cls(n_estimators=n_trees, max_depth=-1 if max_depth is None else max_depth,
                   n_jobs=-1, random_state=seed, verbose=-1).fit(X, y)
    if family == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=n_trees, max_depth=max_depth, n_jobs=-1, random_state=seed).fit(X, y)
    if family == "extra_trees":
        cls = ExtraTreesClassifier if clf else ExtraTreesRegressor
        return cls(n_estimators=n_trees, max_depth=max_depth, n_jobs=-1, random_state=seed).fit(X, y)
    if family == "gradient_boosting":
        cls = GradientBoostingClassifier if clf else GradientBoostingRegressor
        return cls(n_estimators=n_trees, max_depth=max_depth or 3, random_state=seed).fit(X, y)
    if family == "hist_gradient_boosting":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_iter=n_trees, max_depth=max_depth, early_stopping=False,
                   random_state=seed).fit(X, y)
    if family == "decision_tree":
        cls = DecisionTreeClassifier if clf else DecisionTreeRegressor
        return cls(max_depth=max_depth, random_state=seed).fit(X, y)
    raise ValueError(f"unknown model family: {family}")


ALL_FAMILIES = [
    "xgboost", "lightgbm", "random_forest", "extra_trees",
    "gradient_boosting", "hist_gradient_boosting", "decision_tree",
]
SCALING_FAMILIES = ["xgboost", "lightgbm", "random_forest"]   # for the sweep experiments


# ---------------------------------------------------------------------------
# Experiment 1 — model type coverage
# ---------------------------------------------------------------------------

def experiment_model_types(
    datasets: Sequence[str], n_rows: int, n_trees: int, max_depth: int, repeats: int,
) -> pd.DataFrame:
    """Every parser x every model family, on both a regression and a classification set."""
    rows = []
    for ds in datasets:
        X, y, task = load_dataset(ds, n_rows)
        feature_names = list(X.columns)
        print(f"\n=== model_types | {ds} ({task}, {X.shape[0]:,} rows x {X.shape[1]} features) ===")
        for family in ALL_FAMILIES:
            try:
                model = build_model(family, task, X, y, n_trees=n_trees, max_depth=max_depth)
            except Exception as exc:
                print(f"  {family:24s} TRAIN FAILED: {type(exc).__name__}: {exc}")
                continue
            line = [f"  {family:24s}"]
            for parser in ALL_PARSERS:
                res = time_parse(parser, model, feature_names, ds, family, repeats=repeats,
                                 family=family, task=task, n_trees_cfg=n_trees,
                                 max_depth_cfg=max_depth)
                rows.append(res.as_row())
                line.append(f"{parser.name}="
                            + (f"{res.median_s * 1e3:.1f}ms" if res.ok else f"[{res.error_type}]"))
            print(" ".join(line), flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 2 — parse time vs ensemble size
# ---------------------------------------------------------------------------

def experiment_tree_count(
    datasets: Sequence[str], n_rows: int, tree_counts: Sequence[int], max_depth: int,
    repeats: int, budget_s: float = 20.0, retire_s: float = 30.0,
) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        X, y, task = load_dataset(ds, n_rows)
        feature_names = list(X.columns)
        print(f"\n=== tree_count | {ds} ({task}) | depth={max_depth} ===")
        for family in SCALING_FAMILIES:
            retired: set[str] = set()
            for n_trees in sorted(tree_counts):          # ascending, so retiring is safe
                model = build_model(family, task, X, y, n_trees=n_trees, max_depth=max_depth)
                line = [f"  {family:16s} trees={n_trees:>5}"]
                for parser in ALL_PARSERS:
                    cfg: dict[str, Any] = dict(family=family, task=task,
                                               n_trees_cfg=n_trees, max_depth_cfg=max_depth)
                    if parser.name in retired:
                        rows.append(_skipped(parser.name, f"{family}_{n_trees}", ds,
                                             retire_s, **cfg).as_row())
                        line.append(f"{parser.name}=[skipped]")
                        continue
                    res = time_parse(parser, model, feature_names, ds, f"{family}_{n_trees}",
                                     repeats=repeats, budget_s=budget_s, **cfg)
                    rows.append(res.as_row())
                    line.append(f"{parser.name}="
                                + (f"{res.median_s * 1e3:8.1f}ms" if res.ok
                                   else f"[{res.error_type}]"))
                    if res.ok and res.median_s > retire_s:
                        retired.add(parser.name)
                print(" ".join(line), flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 3 — parse time vs depth
# ---------------------------------------------------------------------------

def experiment_depth(
    datasets: Sequence[str], n_rows: int, depths: Sequence[Optional[int]], n_trees: int,
    repeats: int, budget_s: float = 20.0, retire_s: float = 30.0,
) -> pd.DataFrame:
    """
    Depth sweep. ``None`` (unlimited) is always visited last, since it produces the deepest
    trees and is where a slow parser is most likely to be retired.
    """
    rows = []
    ordered = sorted((d for d in depths if d is not None))
    if any(d is None for d in depths):
        ordered = [*ordered, None]

    for ds in datasets:
        X, y, task = load_dataset(ds, n_rows)
        feature_names = list(X.columns)
        print(f"\n=== depth | {ds} ({task}) | trees={n_trees} ===")
        for family in SCALING_FAMILIES:
            retired: set[str] = set()
            for depth in ordered:
                model = build_model(family, task, X, y, n_trees=n_trees, max_depth=depth)
                # "unlimited", not "None": pandas reads a literal "None" back as NaN, which
                # would make the unlimited-depth rows indistinguishable from missing data.
                label = "unlimited" if depth is None else str(depth)
                line = [f"  {family:16s} depth={label:>4}"]
                for parser in ALL_PARSERS:
                    cfg: dict[str, Any] = dict(family=family, task=task,
                                               n_trees_cfg=n_trees, max_depth_cfg=label)
                    if parser.name in retired:
                        rows.append(_skipped(parser.name, f"{family}_d{label}", ds,
                                             retire_s, **cfg).as_row())
                        line.append(f"{parser.name}=[skipped]")
                        continue
                    res = time_parse(parser, model, feature_names, ds, f"{family}_d{label}",
                                     repeats=repeats, budget_s=budget_s, **cfg)
                    rows.append(res.as_row())
                    line.append(f"{parser.name}="
                                + (f"{res.median_s * 1e3:8.1f}ms" if res.ok
                                   else f"[{res.error_type}]"))
                    if res.ok and res.median_s > retire_s:
                        retired.add(parser.name)
                print(" ".join(line), flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correctness: does parser 5 agree with shapiq's native converter?
# ---------------------------------------------------------------------------

# Split semantics differ by library and must be reproduced exactly, otherwise any row whose
# feature value sits on (or very near) a threshold is routed the wrong way. All three
# behaviours below were verified against the libraries' own `.predict()`:
#
#   XGBoost   : `value <  threshold` goes left; input cast to float32
#   sklearn   : `value <= threshold` goes left; input cast to float32
#               (`sklearn.tree._tree.DTYPE` is float32 — `predict` downcasts X, while
#                thresholds stay float64. Ignoring this misroutes rows whose float64 value
#                differs from the threshold but whose float32 value equals it exactly:
#                on California Housing that was every probe row, ~4e-2 of error.)
#   LightGBM  : `value <= threshold` goes left; input stays float64
#
# Thresholds are always compared at their stored precision — never downcast, because
# sklearn's are midpoints of two float32 values and are not float32-representable.
#
# shapiq's TreeModel has no slot for the comparison operator (its TreeSHAP traversal assumes
# `<=`), so this is a property of the source model, not of any converter.
_STRICT_LT_FAMILIES = {"xgboost"}
_FLOAT32_X_FAMILIES = {
    "xgboost", "random_forest", "extra_trees", "gradient_boosting", "decision_tree",
}


def _predict_shapiq_trees(
    trees: Sequence[Any], X: np.ndarray, strict_lt: bool = False, x_float32: bool = False,
) -> np.ndarray:
    """Sum a shapiq ``TreeModel`` list over rows, honouring the source library's semantics."""
    rows = X.astype(np.float32) if x_float32 else X
    out = np.zeros(len(X), dtype=np.float64)
    for tree in trees:
        left, right = tree.children_left, tree.children_right
        feats, thr, vals = tree.features, tree.thresholds, tree.values
        for i, row in enumerate(rows):
            node = 0
            while left[node] != -1:
                value, threshold = row[feats[node]], thr[node]
                go_left = value < threshold if strict_lt else value <= threshold
                node = left[node] if go_left else right[node]
            out[i] += vals[node]
    return out


def _canonical_walk(tree: Any) -> tuple[list[tuple[int, float]], list[float]]:
    """
    Flatten a tree by depth-first traversal from the root, always taking left before right.

    Node *indices* are an implementation detail — treelite and shapiq number LightGBM nodes
    completely differently for the same tree — so comparing the raw arrays reports spurious
    mismatches. A canonical DFS is invariant to renumbering and compares the actual shape.

    Returns
    -------
    (splits, leaves)
        ``splits`` are ``(feature, threshold)`` in DFS order; ``leaves`` are leaf values in
        DFS order.
    """
    splits: list[tuple[int, float]] = []
    leaves: list[float] = []
    left, right = tree.children_left, tree.children_right
    stack = [0]
    while stack:
        node = stack.pop()
        if left[node] == -1:
            leaves.append(float(tree.values[node]))
        else:
            splits.append((int(tree.features[node]), float(tree.thresholds[node])))
            stack.append(right[node])          # pushed first so left is popped first
            stack.append(left[node])
    return splits, leaves


def _tree_structure_diff(ours: Any, native: Any) -> Optional[str]:
    """Compare two shapiq TreeModels up to node renumbering. None when equivalent."""
    if len(ours.children_left) != len(native.children_left):
        return f"node count {len(ours.children_left)} vs {len(native.children_left)}"

    our_splits, our_leaves = _canonical_walk(ours)
    nat_splits, nat_leaves = _canonical_walk(native)

    if len(our_splits) != len(nat_splits) or len(our_leaves) != len(nat_leaves):
        return "different shape (split/leaf counts)"
    if [f for f, _ in our_splits] != [f for f, _ in nat_splits]:
        return "split features differ"

    # Not exact equality: shapiq's XGBoost path reads thresholds from trees_to_dataframe(),
    # which round-trips them through a decimal representation and drops the low bits of the
    # stored float32. The values still agree to well within float32 resolution.
    if not np.allclose([t for _, t in our_splits], [t for _, t in nat_splits],
                       rtol=1e-6, atol=1e-6):
        return "thresholds differ beyond float32 resolution"

    if our_leaves:
        delta = np.asarray(nat_leaves) - np.asarray(our_leaves)
        # A constant per-tree gap is expected: shapiq folds intercept/n_trees into every
        # leaf, treelite keeps the intercept in base_scores. A varying gap is a real bug.
        if float(np.ptp(delta)) > 1e-6 * max(1.0, float(np.abs(delta).max())):
            return f"leaf values differ non-uniformly (spread {np.ptp(delta):.3e})"
    return None


def equivalence_check(
    datasets: Sequence[str], n_rows: int, n_trees: int, max_depth: int, n_probe: int = 200,
) -> pd.DataFrame:
    """
    Validate :func:`treelite_to_shapiq` two independent ways.

    1. **Structural** — every tree is compared field by field against shapiq's own converter
       (:func:`_tree_structure_diff`). This is the strict test and needs no traversal, so it
       cannot be fooled by split-semantics subtleties.
    2. **Predictive** — the converted trees are traversed with the source library's real
       split semantics and checked against ``model.predict``. Leaf-value *conventions* differ
       (shapiq folds the intercept into leaves, treelite keeps it in ``base_scores``), so
       agreement is required only up to a constant offset.

    Both are reported, because they fail in different ways: a structural pass with a
    predictive failure means the split semantics are being misread, and the reverse means
    two different tree sets happen to score alike on the probe rows.
    """
    rows = []
    for ds in datasets:
        X, y, task = load_dataset(ds, n_rows)
        feature_names = list(X.columns)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(n_probe, len(X)), replace=False)
        probe = X.iloc[idx].to_numpy(np.float64)

        for family in ALL_FAMILIES:
            row: dict[str, Any] = {"dataset": ds, "family": family, "task": task}
            try:
                model = build_model(family, task, X, y, n_trees=n_trees, max_depth=max_depth)
            except Exception as exc:
                rows.append({**row, "structural": f"train failed: {type(exc).__name__}"})
                continue

            try:
                ours = TreeliteToShapiqParser().parse(model, feature_names)
            except Exception as exc:
                rows.append({**row, "structural": f"ours failed: {type(exc).__name__}"})
                continue
            row["n_trees_ours"] = len(ours)

            # --- predictive check against the model itself (ground truth) ---
            try:
                truth = np.asarray(model.predict(X.iloc[idx]), dtype=np.float64).ravel()
                pred = _predict_shapiq_trees(
                    ours, probe,
                    strict_lt=family in _STRICT_LT_FAMILIES,
                    x_float32=family in _FLOAT32_X_FAMILIES,
                )
                if task == "classification":
                    # Boosters predict raw margins and forests predict probabilities; both
                    # are monotone in the summed leaf values, so rank agreement is the
                    # meaningful check here rather than absolute error.
                    order_ok = np.array_equal(np.argsort(pred), np.argsort(truth))
                    row["predictive"] = "RANK-MATCH" if order_ok else "RANK-MISMATCH"
                else:
                    resid = truth - pred
                    dev = float(np.max(np.abs(resid - np.mean(resid))))
                    row["pred_max_dev"] = dev
                    row["pred_offset"] = float(np.mean(resid))
                    row["predictive"] = "MATCH" if dev < 1e-4 else "MISMATCH"
            except Exception as exc:
                row["predictive"] = f"error: {type(exc).__name__}"

            # --- structural check against shapiq's own converter ---
            try:
                native = ShapiqParser().parse(model, feature_names)
            except Exception as exc:
                row["structural"] = "SHAPIQ-UNSUPPORTED (ours works)"
                row["shapiq_error"] = type(exc).__name__
                rows.append(row)
                continue

            row["n_trees_native"] = len(native)
            if len(ours) != len(native):
                row["structural"] = f"tree count {len(ours)} vs {len(native)}"
            else:
                diffs = [d for d in (_tree_structure_diff(a, b) for a, b in zip(ours, native)) if d]
                row["structural"] = "MATCH" if not diffs else f"MISMATCH: {diffs[0]} ({len(diffs)} trees)"
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def support_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot of which parser handled which model family."""
    df = df.copy()
    df["cell"] = np.where(df["ok"], "ok", "-- " + df["error_type"])
    return df.pivot_table(index=["dataset", "family"], columns="parser",
                          values="cell", aggfunc="first")


def timing_pivot(df: pd.DataFrame, index: Sequence[str]) -> pd.DataFrame:
    """Median parse time in milliseconds, parsers as columns."""
    ok = df[df["ok"]].copy()
    ok["ms"] = ok["median_s"] * 1e3
    return ok.pivot_table(index=list(index), columns="parser", values="ms", aggfunc="median")


def save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--experiment",
                   choices=["all", "model_types", "tree_count", "depth", "equivalence"],
                   default="all")
    p.add_argument("--datasets", nargs="+", default=["california_housing", "higgs"])
    p.add_argument("--rows", type=int, default=20_000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--quick", action="store_true",
                   help="Tiny sweep for smoke-testing the harness.")
    p.add_argument("--budget-s", type=float, default=20.0,
                   help="Per-configuration time budget; drop repeats once exceeded.")
    p.add_argument("--retire-s", type=float, default=30.0,
                   help="Retire a parser from the rest of a sweep once one parse costs this "
                        "long. Sweeps run ascending, so later points would only be slower.")
    args = p.parse_args(argv)

    tree_counts = [10, 50, 100] if args.quick else [10, 50, 100, 500, 1000]
    depths = [2, 4, 6] if args.quick else [2, 4, 6, 8, 10, 12, None]
    rows = 2_000 if args.quick else args.rows
    repeats = 2 if args.quick else args.repeats
    depth_trees = 20 if args.quick else 100
    type_trees = 20 if args.quick else 100

    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 400)
    want = args.experiment

    if want in ("all", "model_types"):
        df = experiment_model_types(args.datasets, rows, type_trees, 6, repeats)
        save(df, "model_types.csv")
        print("\n--- support matrix (model family x parser) ---")
        print(support_matrix(df).to_string())
        print("\n--- median parse time, ms ---")
        print(timing_pivot(df, ["dataset", "family"]).round(2).to_string())

    if want in ("all", "tree_count"):
        df = experiment_tree_count(args.datasets, rows, tree_counts, 6, repeats,
                                   budget_s=args.budget_s, retire_s=args.retire_s)
        save(df, "tree_count.csv")
        print("\n--- median parse time vs ensemble size, ms ---")
        print(timing_pivot(df, ["dataset", "family", "n_trees_cfg"]).round(2).to_string())

    if want in ("all", "depth"):
        df = experiment_depth(args.datasets, rows, depths, depth_trees, repeats,
                              budget_s=args.budget_s, retire_s=args.retire_s)
        save(df, "depth.csv")
        print("\n--- median parse time vs depth, ms ---")
        print(timing_pivot(df, ["dataset", "family", "max_depth_cfg"]).round(2).to_string())

    if want in ("all", "equivalence"):
        print("\n=== equivalence: treelite->shapiq (ours) vs shapiq native ===")
        df = equivalence_check(args.datasets[:1], rows, type_trees, 6)
        save(df, "equivalence.csv")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
