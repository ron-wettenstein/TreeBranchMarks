"""
Tests for _compute_scores — the core scoring logic in html_generator.py.

Scoring rules:
  - Per (dataset, mission, task, n, m, D, ensemble) group:
      * winner (lowest time) → 100 pts
      * loser  → (winner_time / loser_time) * 100 pts
      * not_supported / memory_crash / runtime_error → 0 pts
  - Groups with fewer than 2 methods are skipped entirely
  - Groups where all valid times are 0 are skipped (division guard)
  - Scores are averaged per mission and overall
"""

import pytest
from treebranchmarks.report.html_generator import _compute_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    method: str,
    running_time: float,
    dataset: str = "ds",
    mission: str = "mission_a",
    task: str = "BG SHAP",
    n: int = 100,
    m: int = 50,
    D: int = 6,
    ensemble: str = "lightgbm",
    not_supported: bool = False,
    memory_crash: bool = False,
    runtime_error: bool = False,
) -> dict:
    return {
        "method": method,
        "running_time": running_time,
        "dataset": dataset,
        "mission": mission,
        "task": task,
        "n": n,
        "m": m,
        "D": D,
        "ensemble": ensemble,
        "not_supported": not_supported,
        "memory_crash": memory_crash,
        "runtime_error": runtime_error,
    }


def _scores(rows):
    return _compute_scores(rows)


# ---------------------------------------------------------------------------
# Basic winner / loser scoring
# ---------------------------------------------------------------------------

def test_winner_gets_100():
    rows = [_row("fast", 1.0), _row("slow", 2.0)]
    result = _scores(rows)
    assert result["overall"]["scores"]["fast"] == pytest.approx(100.0)


def test_loser_gets_proportional_score():
    rows = [_row("fast", 1.0), _row("slow", 4.0)]
    result = _scores(rows)
    assert result["overall"]["scores"]["slow"] == pytest.approx(25.0)


def test_equal_times_both_get_100():
    rows = [_row("a", 2.0), _row("b", 2.0)]
    result = _scores(rows)
    assert result["overall"]["scores"]["a"] == pytest.approx(100.0)
    assert result["overall"]["scores"]["b"] == pytest.approx(100.0)


def test_three_methods_scored_against_winner():
    rows = [_row("a", 1.0), _row("b", 2.0), _row("c", 5.0)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["a"] == pytest.approx(100.0)
    assert scores["b"] == pytest.approx(50.0)
    assert scores["c"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Not-supported / crash → 0 points
# ---------------------------------------------------------------------------

def test_not_supported_gets_zero():
    rows = [_row("fast", 1.0), _row("ns", 0.0, not_supported=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["ns"] == pytest.approx(0.0)
    assert scores["fast"] == pytest.approx(100.0)


def test_memory_crash_gets_zero():
    rows = [_row("fast", 1.0), _row("crash", 0.0, memory_crash=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["crash"] == pytest.approx(0.0)


def test_runtime_error_gets_zero():
    rows = [_row("fast", 1.0), _row("err", 0.0, runtime_error=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["err"] == pytest.approx(0.0)


def test_not_supported_counts_toward_group_size():
    # Group has fast + not_supported → 2 methods → group is valid
    rows = [_row("fast", 1.0), _row("ns", 0.0, not_supported=True)]
    result = _scores(rows)
    assert result["overall"] is not None


# ---------------------------------------------------------------------------
# Groups skipped when fewer than 2 methods
# ---------------------------------------------------------------------------

def test_single_method_group_skipped():
    rows = [_row("only", 1.0)]
    result = _scores(rows)
    assert result["overall"] is None


def test_single_method_not_supported_skipped():
    rows = [_row("only", 0.0, not_supported=True)]
    result = _scores(rows)
    assert result["overall"] is None


def test_two_methods_in_different_groups_each_skipped():
    # "a" only appears in group with n=100, "b" only in group with n=200 → both skipped
    rows = [
        _row("a", 1.0, n=100),
        _row("b", 2.0, n=200),
    ]
    result = _scores(rows)
    assert result["overall"] is None


# ---------------------------------------------------------------------------
# Groups are keyed by (dataset, mission, task, n, m, D, ensemble)
# ---------------------------------------------------------------------------

def test_different_D_different_groups():
    rows = [
        _row("a", 1.0, D=6), _row("b", 4.0, D=6),
        _row("a", 3.0, D=12), _row("b", 6.0, D=12),
    ]
    result = _scores(rows)
    # 2 groups × 2 methods = 2 runs → overall average
    scores = result["overall"]["scores"]
    # D=6: a=100, b=25; D=12: a=100, b=50 → averages: a=100, b=37.5
    assert scores["a"] == pytest.approx(100.0)
    assert scores["b"] == pytest.approx(37.5)
    assert result["overall"]["n"] == 2


def test_different_n_different_groups():
    rows = [
        _row("a", 1.0, n=100), _row("b", 4.0, n=100),
        _row("a", 2.0, n=200), _row("b", 2.0, n=200),
    ]
    scores = _scores(rows)["overall"]["scores"]
    # n=100: a=100, b=25; n=200: a=100, b=100 → avg: a=100, b=62.5
    assert scores["a"] == pytest.approx(100.0)
    assert scores["b"] == pytest.approx(62.5)


def test_different_dataset_different_groups():
    rows = [
        _row("a", 1.0, dataset="ds1"), _row("b", 2.0, dataset="ds1"),
        _row("a", 1.0, dataset="ds2"), _row("b", 2.0, dataset="ds2"),
    ]
    result = _scores(rows)
    assert result["overall"]["n"] == 2


# ---------------------------------------------------------------------------
# by_mission aggregation
# ---------------------------------------------------------------------------

def test_by_mission_keys():
    rows = [
        _row("a", 1.0, mission="m1"), _row("b", 2.0, mission="m1"),
        _row("a", 1.0, mission="m2"), _row("b", 3.0, mission="m2"),
    ]
    result = _scores(rows)
    assert "m1" in result["by_mission"]
    assert "m2" in result["by_mission"]


def test_by_mission_scores_independent():
    rows = [
        _row("a", 1.0, mission="m1"), _row("b", 2.0, mission="m1"),
        _row("a", 1.0, mission="m2"), _row("b", 10.0, mission="m2"),
    ]
    result = _scores(rows)
    # m1: a=100, b=50
    assert result["by_mission"]["m1"]["scores"]["b"] == pytest.approx(50.0)
    # m2: a=100, b=10
    assert result["by_mission"]["m2"]["scores"]["b"] == pytest.approx(10.0)


def test_overall_is_average_across_missions():
    rows = [
        _row("a", 1.0, mission="m1"), _row("b", 2.0, mission="m1"),  # b=50
        _row("a", 1.0, mission="m2"), _row("b", 4.0, mission="m2"),  # b=25
    ]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["b"] == pytest.approx(37.5)  # (50 + 25) / 2


def test_overall_n_counts_groups_not_rows():
    rows = [
        _row("a", 1.0, mission="m1"), _row("b", 2.0, mission="m1"),
        _row("a", 1.0, mission="m2"), _row("b", 2.0, mission="m2"),
    ]
    result = _scores(rows)
    assert result["overall"]["n"] == 2


# ---------------------------------------------------------------------------
# methods list
# ---------------------------------------------------------------------------

def test_methods_list_sorted():
    rows = [_row("z", 1.0), _row("a", 2.0)]
    result = _scores(rows)
    assert result["methods"] == ["a", "z"]


def test_methods_list_excludes_not_in_any_group():
    # Single method → group skipped → appears in no run → not in methods list
    rows = [_row("lonely", 1.0)]
    result = _scores(rows)
    assert result["methods"] == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_rows():
    result = _scores([])
    assert result["overall"] is None
    assert result["by_mission"] == {}
    assert result["methods"] == []


def test_row_with_no_method_field_skipped():
    rows = [{"running_time": 1.0, "dataset": "ds", "mission": "m", "task": "t",
             "n": 100, "m": 0, "D": 6, "ensemble": "lgbm", "method": ""}]
    result = _scores(rows)
    assert result["overall"] is None


def test_all_zero_times_produces_empty_scores():
    # Both methods have time=0 → no supported times → group registers but scores are empty
    rows = [_row("a", 0.0), _row("b", 0.0)]
    result = _scores(rows)
    assert result["overall"]["scores"] == {}


def test_both_crashed_get_zero():
    # Both memory_crash → both get 0 pts
    rows = [_row("a", 0.0, memory_crash=True), _row("b", 0.0, memory_crash=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.0)


def test_both_not_supported_get_zero():
    rows = [_row("a", 0.0, not_supported=True), _row("b", 0.0, not_supported=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.0)


def test_both_runtime_error_get_zero():
    rows = [_row("a", 0.0, runtime_error=True), _row("b", 0.0, runtime_error=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# One crashes, one valid — winner is the valid one
# ---------------------------------------------------------------------------

def test_memory_crash_vs_valid():
    rows = [_row("fast", 2.0), _row("crash", 0.0, memory_crash=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["fast"] == pytest.approx(100.0)
    assert scores["crash"] == pytest.approx(0.0)


def test_runtime_error_vs_valid():
    rows = [_row("fast", 2.0), _row("err", 0.0, runtime_error=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["fast"] == pytest.approx(100.0)
    assert scores["err"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mixed crash types in the same group
# ---------------------------------------------------------------------------

def test_memory_crash_vs_not_supported():
    # Both are flagged → both get 0, group is valid (2 methods)
    rows = [_row("crash", 0.0, memory_crash=True), _row("ns", 0.0, not_supported=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["crash"] == pytest.approx(0.0)
    assert scores["ns"] == pytest.approx(0.0)


def test_three_methods_one_crashes():
    rows = [_row("a", 1.0), _row("b", 4.0), _row("crash", 0.0, memory_crash=True)]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["a"] == pytest.approx(100.0)
    assert scores["b"] == pytest.approx(25.0)
    assert scores["crash"] == pytest.approx(0.0)


def test_three_methods_two_crash():
    rows = [
        _row("a", 2.0),
        _row("crash1", 0.0, memory_crash=True),
        _row("crash2", 0.0, not_supported=True),
    ]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["a"] == pytest.approx(100.0)
    assert scores["crash1"] == pytest.approx(0.0)
    assert scores["crash2"] == pytest.approx(0.0)


def test_one_zero_one_valid_time():
    # a=0 is filtered out from supported; b=1.0 is the only supported method
    # Only 1 supported method but there are 2 methods total in the group
    # → group valid (2 total), but only b gets a real score; a gets 0 via t>0 filter
    rows = [_row("a", 0.0), _row("b", 1.0)]
    result = _scores(rows)
    # Group has 2 methods, b=1.0 is the winner → b=100; a is filtered from supported → no score
    scores = result["overall"]["scores"]
    assert scores["b"] == pytest.approx(100.0)
    assert "a" not in scores  # a had time=0, not in supported, not flagged as not_supported either


def test_mixed_not_supported_and_valid():
    rows = [
        _row("fast", 1.0),
        _row("medium", 2.0),
        _row("ns", 0.0, not_supported=True),
    ]
    scores = _scores(rows)["overall"]["scores"]
    assert scores["fast"] == pytest.approx(100.0)
    assert scores["medium"] == pytest.approx(50.0)
    assert scores["ns"] == pytest.approx(0.0)
