"""Tests for Task, TaskResult, and ApproachResult."""

import pytest
from pathlib import Path

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.method import Method
from treebranchmarks.core.task import Task, TaskType, TaskResult, ApproachResult
from treebranchmarks.core.params import TreeParameters, EnsembleType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_approach(name: str) -> Approach:
    class _A(Approach):
        pass
    _A.name = name
    _A.method = Method(name="test", label="Test")
    return _A()


def _make_params() -> TreeParameters:
    return TreeParameters(T=10, D=6, L=32.0, F=20, ensemble_type=EnsembleType.LIGHTGBM, n=100, m=0)


# ---------------------------------------------------------------------------
# Task construction — duplicate name check
# ---------------------------------------------------------------------------

def test_task_raises_on_duplicate_names():
    a1 = _make_approach("shap")
    a2 = _make_approach("shap")
    with pytest.raises(ValueError, match="Duplicate approach names"):
        Task(TaskType.PATH_DEPENDENT_SHAP, [a1, a2])


def test_task_unique_names_ok():
    a1 = _make_approach("shap")
    a2 = _make_approach("woodelf")
    task = Task(TaskType.PATH_DEPENDENT_SHAP, [a1, a2])
    assert len(task.approaches) == 2


def test_task_empty_approaches_ok():
    task = Task(TaskType.BACKGROUND_SHAP, [])
    assert task.approaches == []

# ---------------------------------------------------------------------------
# ApproachResult serialization
# ---------------------------------------------------------------------------

def test_approach_result_as_dict_round_trip():
    r = ApproachResult(
        approach_name="shap",
        running_time=1.23,
        std_time_s=0.01,
        is_estimated=False,
        error=None,
        method="shap",
        not_supported=False,
        memory_crash=False,
        runtime_error=False,
    )
    d = r.as_dict()
    assert d["approach_name"] == "shap"
    assert d["running_time"] == 1.23
    assert d["std_time_s"] == 0.01
    assert d["is_estimated"] is False
    assert d["error"] is None
    assert d["method"] == "shap"
    assert d["not_supported"] is False
    assert d["memory_crash"] is False
    assert d["runtime_error"] is False


def test_approach_result_as_dict_with_error():
    r = ApproachResult(
        approach_name="broken",
        running_time=0.0,
        std_time_s=0.0,
        is_estimated=False,
        error="Traceback: something went wrong",
        method="broken",
        runtime_error=True,
    )
    d = r.as_dict()
    assert d["runtime_error"] is True
    assert "something went wrong" in d["error"]


# ---------------------------------------------------------------------------
# TaskResult serialization
# ---------------------------------------------------------------------------

def test_task_result_as_dict():
    r = ApproachResult(
        approach_name="shap",
        running_time=2.0,
        std_time_s=0.0,
        is_estimated=False,
        error=None,
        method="shap",
    )
    params = _make_params()
    tr = TaskResult(
        task_name="Background SHAP",
        params=params,
        approach_results={"shap": r},
    )
    d = tr.as_dict()
    assert d["task_name"] == "Background SHAP"
    assert "shap" in d["approach_results"]
    assert d["approach_results"]["shap"]["running_time"] == 2.0
    assert d["params"]["D"] == 6
    assert d["params"]["ensemble_type"] == "lightgbm"
