"""
Parity between the two scoring implementations.

The HTML report computes scores twice: once in Python (``_compute_scores``,
embedded as the ``SCORES`` constant) and once in the browser
(``report_scoring.js``'s ``computeScores``), which recomputes them over whatever
subset of methods the global Methods filter has selected.  With every method
checked the two must agree *exactly* — otherwise the Score Summary would jump
the moment a user touched the filter.

These tests run the real JS through node and compare it against Python on
synthetic edge cases and, when they are present, on real report data for every
possible subset of methods.
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from treebranchmarks.report.html_generator import _compute_scores

_REPO = Path(__file__).resolve().parent.parent
_SCORING_JS = _REPO / "treebranchmarks" / "report" / "report_scoring.js"
_RESULTS = _REPO / "results"

TOL = 1e-9

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the JS scorer"
)

_BRIDGE = """
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const computeScores = new Function(src + ';return computeScores;')();
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
fs.writeFileSync(process.argv[4],
  JSON.stringify(cases.map(c => computeScores(c.rows, c.opts || {}))));
"""


def _run_js(cases: list[dict]) -> list[dict]:
    """Score each case with the real report_scoring.js and return the results."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "bridge.js").write_text(_BRIDGE, encoding="utf-8")
        (d / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
        subprocess.run(
            ["node", str(d / "bridge.js"), str(_SCORING_JS),
             str(d / "cases.json"), str(d / "out.json")],
            check=True, capture_output=True,
        )
        return json.loads((d / "out.json").read_text(encoding="utf-8"))


def _assert_same(py, js, path="") -> None:
    """Assert two score trees match, comparing floats within TOL."""
    if py is None or js is None:
        assert py == js, f"{path}: python={py!r} js={js!r}"
        return
    if isinstance(py, dict):
        assert set(py) == set(js), (
            f"{path}: key mismatch python={sorted(py)} js={sorted(js)}"
        )
        for k in py:
            _assert_same(py[k], js[k], f"{path}.{k}")
        return
    if isinstance(py, (int, float)) and isinstance(js, (int, float)):
        assert abs(py - js) <= TOL, f"{path}: python={py!r} js={js!r}"
        return
    assert py == js, f"{path}: python={py!r} js={js!r}"


def _row(approach, running_time, **kw):
    row = {
        "approach": approach, "method": approach, "running_time": running_time,
        "dataset": "ds", "mission": "m1", "task": "t1",
        "n": 100, "m": 50, "D": 6, "T": 100, "ensemble": "lightgbm",
        "not_supported": False, "memory_crash": False, "runtime_error": False,
    }
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------

SYNTHETIC: dict[str, list[dict]] = {
    "empty": [],
    "simple_two": [_row("a", 1.0), _row("b", 2.0)],
    "three_methods": [_row("a", 1.0), _row("b", 2.0), _row("c", 5.0)],
    "single_method_skipped": [_row("a", 1.0)],
    "crash_scores_zero": [_row("a", 1.0), _row("b", 0.0, memory_crash=True)],
    "all_crashed": [
        _row("a", 0.0, memory_crash=True), _row("b", 0.0, not_supported=True),
    ],
    # A zero (non-crash) time counts toward group size but earns no score.
    "zero_time_counts_toward_group": [_row("a", 0.0), _row("b", 1.0)],
    "all_zero_times": [_row("a", 0.0), _row("b", 0.0)],
    # Repeated rows for one approach are averaged before comparing.
    "repeats_are_averaged": [_row("a", 1.0), _row("a", 3.0), _row("b", 4.0)],
    # T is part of the group key — these are two groups, not one.
    "T_splits_groups": [
        _row("a", 1.0, T=100), _row("b", 4.0, T=100),
        _row("a", 3.0, T=500), _row("b", 6.0, T=500),
    ],
    "missions_scored_separately": [
        _row("a", 1.0, mission="m1"), _row("b", 2.0, mission="m1"),
        _row("a", 1.0, mission="m2"), _row("b", 10.0, mission="m2"),
    ],
    "blank_approach_skipped": [
        _row("", 1.0), _row("b", 2.0), _row("c", 4.0),
    ],
}


def test_synthetic_cases_match_python():
    names = sorted(SYNTHETIC)
    cases = [{"rows": SYNTHETIC[n]} for n in names]
    js_results = _run_js(cases)
    for name, js in zip(names, js_results):
        _assert_same(_compute_scores(SYNTHETIC[name]), js, name)


# ---------------------------------------------------------------------------
# Real report data — every subset of methods
# ---------------------------------------------------------------------------

_FIXTURES = [
    "base_woodelf_progress_experiment",
    "model_parsing_v1",
    "woodelf_vs_shapiq_experiment",
    "base_woodelf_progress_experiment_pd_only",
]


def _load_fixture(name: str) -> list[dict] | None:
    """Pull the embedded DATA array out of a previously generated report."""
    import re

    path = _RESULTS / f"{name}.html"
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"const DATA = (\[.*?\]);\n", txt, re.S)
    return json.loads(match.group(1)) if match else None


@pytest.mark.parametrize("fixture", _FIXTURES)
def test_every_method_subset_matches_python(fixture):
    """
    The global Methods filter can select any subset of methods.  For each one,
    the JS scorer must reproduce Python exactly — this is what keeps the Score
    Summary honest when a method is unchecked.
    """
    data = _load_fixture(fixture)
    if data is None:
        pytest.skip(f"results/{fixture}.html not present")

    methods = _compute_scores(data)["methods"]
    subsets = [
        sub
        for k in range(2, len(methods) + 1)
        for sub in itertools.combinations(methods, k)
    ]

    rowsets = [[r for r in data if r.get("approach") in set(sub)] for sub in subsets]
    js_results = _run_js([{"rows": rows} for rows in rowsets])

    for sub, rows, js in zip(subsets, rowsets, js_results):
        _assert_same(_compute_scores(rows), js, f"{fixture}[{'+'.join(sub)}]")


def test_remove_fast_drops_only_all_fast_groups():
    """removeFast is JS-only (the Filtered Score box); Python has no counterpart."""
    rows = [
        # all under 10 s -> dropped when removeFast is on
        _row("a", 1.0, n=100), _row("b", 2.0, n=100),
        # one over 10 s -> kept
        _row("a", 5.0, n=200), _row("b", 50.0, n=200),
    ]
    plain, fast = _run_js([
        {"rows": rows},
        {"rows": rows, "opts": {"removeFast": True}},
    ])
    _assert_same(_compute_scores(rows), plain, "removeFast_off")
    assert plain["overall"]["n"] == 2
    assert fast["overall"]["n"] == 1
    assert fast["overall"]["scores"]["b"] == pytest.approx(10.0)
