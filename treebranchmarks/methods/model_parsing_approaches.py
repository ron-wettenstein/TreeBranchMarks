"""
Model-parsing approaches — the parsers from :mod:`model_parsing` wired into the
Experiment / Mission / Task framework.

Every TreeSHAP-style explainer starts by *parsing* the model: walking the native
booster/forest and materialising a library-specific tree structure. That cost is paid
before a single Shapley value is computed, and again on every explainer construction.
:data:`TaskType.MODEL_PARSING` measures it.

This module is a thin adapter. The parsers themselves — including the treelite -> shapiq
converter — live in :mod:`treebranchmarks.methods.model_parsing` and are imported, not
re-implemented, so the standalone benchmark and this experiment always measure identical
code.

Approaches
----------
============================  ===========================================================
``TreeliteParsing``           treelite front-end (compiler IR, not an explainer format)
``ShapParsing``               ``shap.explainers._tree.TreeEnsemble``
``ShapiqParsing``             ``shapiq...validate_tree_model``
``WoodelfShapParsing``        woodelf via ``parse_models_using_shap``
``WoodelfTreeliteParsing``    woodelf via ``parse_models_using_treelite``
``TreeliteToShapiqParsing``   treelite parse -> shapiq ``TreeModel`` (written in-repo)
============================  ===========================================================

Each is its own :class:`Method`, so the report scores all six against each other rather
than collapsing them into a shap-vs-woodelf pair.

Unsupported combinations report ``not_supported=True`` rather than raising, which is how
the framework records a real capability gap (shapiq cannot parse ``GradientBoosting*`` or
``HistGradientBoosting*``; treelite cannot parse a bare ``DecisionTree``).
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.method import Method
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.model_parsing import (
    ModelParser,
    ShapParser,
    ShapiqParser,
    TreeliteParser,
    TreeliteToShapiqParser,
    WoodelfShapParser,
    WoodelfTreeliteParser,
)

# ---------------------------------------------------------------------------
# Methods — one per parser, so all six are scored against each other
# ---------------------------------------------------------------------------

TREELITE = Method(
    name="treelite", label="treelite",
    description="treelite front-end. A compiler IR rather than an explainer format, "
                "included as a lower bound on what parsing can cost.",
)
SHAP_PARSE = Method(
    name="shap_parse", label="shap",
    description="shap.explainers._tree.TreeEnsemble — the parsing half of TreeExplainer.",
)
SHAPIQ_PARSE = Method(
    name="shapiq_parse", label="shapiq",
    description="shapiq.explainer.tree.validation.validate_tree_model, which dispatches "
                "to shapiq's per-library converters.",
)
WOODELF_SHAP_PARSE = Method(
    name="woodelf_shap_parse", label="woodelf (via shap)",
    description="woodelf.core.trees.parse_models_using_shap.load_model_using_shap — woodelf's "
                "DecisionTreesEnsemble built on top of a shap parse.",
)
WOODELF_TREELITE_PARSE = Method(
    name="woodelf_treelite_parse", label="woodelf (via treelite)",
    description="woodelf.core.trees.parse_models_using_treelite.load_model_using_treelite — "
                "the same ensemble built on top of a treelite parse.",
)
TREELITE_SHAPIQ = Method(
    name="treelite_shapiq", label="treelite -> shapiq",
    description="Written in this repo: parse with treelite, then convert treelite's arrays "
                "into shapiq's TreeModel format. Faster than shapiq's own converter on "
                "boosters, and parses models shapiq rejects.",
)


# ---------------------------------------------------------------------------
# Approach adapter
# ---------------------------------------------------------------------------

class ParseTooSlowError(RuntimeError):
    """
    A parser is refused a model at least as large as one that already blew its budget.

    Raised rather than returned so the framework classifies it as a runtime error, which
    scores 0 — the same treatment any other failure gets. Returning ``not_supported`` would
    be wrong: the parser *can* handle the model, it is simply too slow to be usable.
    """


class ModelParsingApproach(Approach):
    """
    Times one :class:`ModelParser` under :data:`TaskType.MODEL_PARSING`.

    Failure handling distinguishes two genuinely different things:

    * **Cannot parse** -> ``not_supported=True``. A permanent library limitation, e.g.
      shapiq rejecting ``GradientBoosting*``, or treelite rejecting a bare ``DecisionTree``.
    * **Too slow to be usable** -> :class:`ParseTooSlowError`. shapiq needs ~52 s for one
      depth-12 XGBoost model and grows ~3.5x per two levels of depth, so the deepest
      configurations would otherwise dominate the entire experiment's wall clock.

    The budget can only be enforced *after* the fact — there is no way to know a parse is
    slow without running it. So the first configuration to exceed ``BUDGET_S`` is measured
    and reported honestly, and its size is remembered per ensemble type; any later model at
    least that large is refused outright. Missions sweep size ascending, so a refusal always
    concerns a model no smaller than the one that was actually timed.
    """

    #: Wall-clock seconds above which a parser is retired for larger models of this type.
    BUDGET_S = 20.0

    _parser: ModelParser

    def __init__(self) -> None:
        self.parser = self._parser
        # ensemble_type -> (size proxy that blew the budget, seconds it took)
        self._too_slow_at: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _size(trained_model: TrainedModel) -> float:
        """Model size proxy: trees x average leaves, monotone in total node count."""
        params = trained_model.params
        return float(params.T) * float(params.L)

    def model_parsing(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        params = trained_model.params
        ensemble = params.ensemble_type.value
        size = self._size(trained_model)

        previous = self._too_slow_at.get(ensemble)
        if previous is not None and size >= previous[0]:
            prior_size, prior_seconds = previous
            raise ParseTooSlowError(
                f"{self.name}: took {prior_seconds:.1f}s to parse a {ensemble} model of "
                f"size T*L={prior_size:,.0f}, over the {self.BUDGET_S:.0f}s budget. "
                f"This model (T={params.T}, D={params.D}, T*L={size:,.0f}) is at least as "
                f"large, so it was not attempted."
            )

        feature_names = list(X_explain.columns)
        try:
            t0 = time.perf_counter()
            self.parser.parse(trained_model.raw_model, feature_names)
            elapsed = time.perf_counter() - t0
        except ParseTooSlowError:
            raise
        except Exception as exc:
            return ApproachOutput(
                elapsed_s=0.0,
                not_supported=True,
                estimation_description=f"{type(exc).__name__}: "
                                       f"{(str(exc).splitlines() or [''])[0][:160]}",
            )

        if elapsed > self.BUDGET_S:
            best = self._too_slow_at.get(ensemble)
            if best is None or size < best[0]:
                self._too_slow_at[ensemble] = (size, elapsed)

        return ApproachOutput(elapsed_s=elapsed)


class TreeliteParsing(ModelParsingApproach):
    name = "treelite"
    method = TREELITE
    description = TREELITE.description
    _parser = TreeliteParser()


class ShapParsing(ModelParsingApproach):
    name = "shap"
    method = SHAP_PARSE
    description = SHAP_PARSE.description
    _parser = ShapParser()


class ShapiqParsing(ModelParsingApproach):
    name = "shapiq"
    method = SHAPIQ_PARSE
    description = SHAPIQ_PARSE.description
    _parser = ShapiqParser()


class WoodelfShapParsing(ModelParsingApproach):
    name = "woodelf (shap)"
    method = WOODELF_SHAP_PARSE
    description = WOODELF_SHAP_PARSE.description
    _parser = WoodelfShapParser()


class WoodelfTreeliteParsing(ModelParsingApproach):
    name = "woodelf (treelite)"
    method = WOODELF_TREELITE_PARSE
    description = WOODELF_TREELITE_PARSE.description
    _parser = WoodelfTreeliteParser()


class TreeliteToShapiqParsing(ModelParsingApproach):
    name = "treelite -> shapiq"
    method = TREELITE_SHAPIQ
    description = TREELITE_SHAPIQ.description
    _parser = TreeliteToShapiqParser()


ALL_PARSING_APPROACHES = [
    TreeliteParsing, ShapParsing, ShapiqParsing,
    WoodelfShapParsing, WoodelfTreeliteParsing, TreeliteToShapiqParsing,
]


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def ModelParsingTask(
    n_repeats: int = 5,
    extra_approaches: Optional[list[Approach]] = None,
    **task_kwargs,
) -> Task:
    """
    Build the model-parsing Task with all six parsers.

    ``n_repeats`` defaults to 5 because parses are milliseconds long and noisy at n=1.
    Task stops repeating on its own once a single run exceeds 10 s, which is what keeps
    the deepest shapiq configurations from dominating the wall clock.
    """
    approaches: list[Approach] = [cls() for cls in ALL_PARSING_APPROACHES]
    approaches.extend(extra_approaches or [])
    return Task(TaskType.MODEL_PARSING, approaches, n_repeats=n_repeats, **task_kwargs)
