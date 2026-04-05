"""
Interactive HTML report — single-graph view.

All benchmark data is embedded as a JSON array.  Pure JavaScript handles
filtering and drives Plotly.react() so the page needs no server.

Controls
--------
  Dataset  — pick one dataset
  Task     — pick one task (filtered to those present in the selected dataset)
  Mission  — pick one mission (filtered to those present in dataset+task)

The x-axis is automatically determined from the selected mission: whichever of
n / m / D has more than one distinct value in that mission's data is used.
If none (or multiple) vary, defaults to n.

Below the chart a table is rendered: rows = approaches, columns = x-axis tick
values.  Estimated times are shown with a trailing *.
"""

from __future__ import annotations

import json
from pathlib import Path

from treebranchmarks.core.experiment import ExperimentResult
from treebranchmarks.core.method import Method


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _collect_rows(experiment: ExperimentResult) -> list[dict]:
    """Flatten ExperimentResult into a list of plain dicts for JS consumption."""
    rows = []
    for mr in experiment.mission_results:
        if mr.config is not None:
            dataset_name = mr.config.dataset.name
        else:
            dataset_name = getattr(mr, "_dataset_name", "unknown")
        mission_name = getattr(mr, "mission_name", "unknown")

        for tr in mr.task_results:
            p = tr.params
            for approach_name, ar in tr.approach_results.items():
                # Skip plain errors; include classified crashes (memory_crash, runtime_error)
                if ar.error and not ar.runtime_error:
                    continue
                rows.append({
                    "dataset": dataset_name,
                    "mission": mission_name,
                    "task": tr.task_name,
                    "approach": approach_name,
                    "method": ar.method,
                    "n": p.n,
                    "m": p.m,
                    "D": p.D,
                    "T": p.T,
                    "L": round(p.L, 2),
                    "F": p.F,
                    "ensemble": p.ensemble_type.value,
                    "running_time": ar.running_time,
                    "std_s": ar.std_time_s,
                    "is_estimated": ar.is_estimated,
                    "estimation_description": ar.estimation_description,
                    "not_supported": ar.not_supported,
                    "memory_crash": ar.memory_crash,
                    "runtime_error": ar.runtime_error,
                })
    return rows


def _collect_methods(experiment: ExperimentResult) -> list[dict]:
    """
    Return a list of {name, label, description} dicts for every Method found
    in the experiment's approach results, ordered by first appearance.
    Preserves full Method metadata so the JS never needs to hardcode labels.
    """
    seen: dict[str, dict] = {}
    for mr in experiment.mission_results:
        for tr in mr.task_results:
            for ar in tr.approach_results.values():
                if ar.method and ar.method not in seen:
                    seen[ar.method] = {"name": ar.method, "label": ar.method, "description": ""}

    # Overlay rich metadata from actual Method objects when available
    try:
        from treebranchmarks.methods.builtin import SHAP, WOODELF
        for m in (SHAP, WOODELF):
            if m.name in seen:
                seen[m.name] = m.as_dict()
    except ImportError:
        pass

    return list(seen.values())


def _collect_mission_meta(experiment: ExperimentResult) -> dict:
    """Build a dict keyed by mission_name → metadata for the details panel."""
    meta: dict = {}
    for mr in experiment.mission_results:
        mission_name = getattr(mr, "mission_name", "unknown")
        if mission_name not in meta:
            meta[mission_name] = getattr(mr, "meta", {})
    return meta


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_scores(rows: list[dict]) -> dict:
    """
    Compare all methods against each other for each unique
    (dataset, mission, task, n, m, D, ensemble) group.

    Scoring rule per group:
      - winner (lowest time) gets 100
      - each other method gets (winner_time / their_time) * 100
      - groups where fewer than 2 methods have valid (>0) times are skipped

    Returns a dict ready for JSON embedding:
      {
        "methods": ["shap", "woodelf", ...],   # all method names found
        "overall":    { "scores": {"shap": 87.3, "woodelf": 54.2}, "n": 15 },
        "by_mission": { "mission_name": { "scores": {...}, "n": 5 }, ... }
      }
    """
    from collections import defaultdict

    # group_data[key] = { "_times": {method: [times]}, "_not_supported": {methods}, "mission": str, "task": str }
    group_data: dict = defaultdict(lambda: {"_times": defaultdict(list), "_not_supported": set(), "mission": "", "task": ""})

    for r in rows:
        method = r.get("method", "")
        if not method:
            continue
        key = (r["dataset"], r["mission"], r["task"], r["n"], r["m"], r["D"], r["ensemble"])
        g = group_data[key]
        if r.get("not_supported") or r.get("memory_crash") or r.get("runtime_error"):
            g["_not_supported"].add(method)
        else:
            g["_times"][method].append(r["running_time"])
        g["mission"] = r["mission"]
        g["task"] = r["task"]

    # runs[i] = { "mission": str, "task": str, "method_scores": {method: score} }
    runs: list[dict] = []
    all_methods: set[str] = set()

    for g in group_data.values():
        times_by_method = {m: sum(ts) / len(ts) for m, ts in g["_times"].items() if ts}
        supported = {m: t for m, t in times_by_method.items() if t > 0}
        not_supported_methods = g["_not_supported"]
        # Need at least 2 methods total (supported or crashed) to form a comparison
        all_methods_in_group = set(times_by_method.keys()) | not_supported_methods
        if len(all_methods_in_group) < 2:
            continue
        if supported:
            winner_time = min(supported.values())
            scores = {m: (winner_time / t) * 100.0 for m, t in supported.items()}
        else:
            scores = {}
        for m in not_supported_methods:
            scores[m] = 0.0
        all_methods.update(scores.keys())
        runs.append({"mission": g["mission"], "task": g["task"], "method_scores": scores})

    def avg_scores(subset: list) -> dict | None:
        if not subset:
            return None
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for r in subset:
            for m, s in r["method_scores"].items():
                totals[m] = totals.get(m, 0.0) + s
                counts[m] = counts.get(m, 0) + 1
        return {
            "scores": {m: totals[m] / counts[m] for m in totals},
            "n": len(subset),
        }

    by_mission: dict = {}
    for mn in {r["mission"] for r in runs}:
        by_mission[mn] = avg_scores([r for r in runs if r["mission"] == mn])

    return {
        "methods": sorted(all_methods),
        "overall": avg_scores(runs),
        "by_mission": by_mission,
    }


# ---------------------------------------------------------------------------
# HtmlGenerator
# ---------------------------------------------------------------------------

class HtmlGenerator:
    """
    Generates a self-contained interactive HTML report from an ExperimentResult.

    Usage
    -----
    >>> gen = HtmlGenerator()
    >>> gen.generate(result, Path("results/report.html"))
    """

    def generate(
        self,
        result: ExperimentResult,
        output_path: Path,
        summary_html: str | None = None,
    ) -> None:
        rows = _collect_rows(result)
        if not rows:
            raise ValueError("ExperimentResult contains no successful approach results.")

        data_js    = json.dumps(rows, separators=(",", ":"))
        meta_js    = json.dumps(_collect_mission_meta(result), separators=(",", ":"))
        scores_js  = json.dumps(_compute_scores(rows), separators=(",", ":"))
        methods_js = json.dumps(_collect_methods(result), separators=(",", ":"))
        html = _build_html(result.experiment_name, data_js, meta_js, scores_js, methods_js, summary_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML assembly — use concatenation so JS braces need no escaping
# ---------------------------------------------------------------------------

_CSS_PATH       = Path(__file__).parent / "report.css"
_JS_PATH        = Path(__file__).parent / "report.js"
_FRAMEWORK_PATH = Path(__file__).parent / "framework_summary.html"


def _details_panel(title: str, content: str) -> str:
    """Render a closed collapsible <details> panel with summary-panel styling."""
    return (
        "  <details class=\"details-panel summary-panel\">\n"
        f"    <summary>{title}</summary>\n"
        "    <div class=\"summary-content\">\n"
        + content + "\n"
        "    </div>\n"
        "  </details>\n"
    )


def _summary_html(content: str) -> str:
    return _details_panel("About this Experiment", content)


def _framework_html() -> str:
    return _details_panel("How to Use This Report", _FRAMEWORK_PATH.read_text(encoding="utf-8"))


def _build_html(experiment_name: str, data_js: str, meta_js: str, scores_js: str, methods_js: str, summary_html: str | None = None) -> str:
    css = _CSS_PATH.read_text(encoding="utf-8")
    js  = _JS_PATH.read_text(encoding="utf-8")
    head = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>TreeBranchMarks \u2014 {experiment_name}</title>\n"
        "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n"
        "  <style>\n" + css + "\n  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>TreeBranchMarks</h1>\n"
        f"  <div class=\"subtitle\">Experiment: <strong>{experiment_name}</strong></div>\n"
        + (_summary_html(summary_html) if summary_html is not None else "")
        + _framework_html()
        + "  <div id=\"scoreboard\"></div>\n"
        "  <h2 class=\"all-results-heading\">Experiment Analysis</h2>\n"
        "  <div id=\"experiment-panel\">\n"
        + _controls_html() +
        "    <details id=\"mission-details\" class=\"details-panel\">\n"
        "      <summary>Details: dataset, model &amp; approaches</summary>\n"
        "      <div id=\"mission-details-content\"></div>\n"
        "    </details>\n"
        "    <div id=\"mission-score-banner\"></div>\n"
        "    <div id=\"chart\"></div>\n"
        "    <div id=\"data-table\"></div>\n"
        "  </div>\n"
        "  <h2 class=\"analytics-heading\">Analytics</h2>\n"
        "  <div id=\"analytics-section\">\n"
        "    <div class=\"ar-filters\" id=\"ana-filters\"></div>\n"
        "    <div id=\"ana-score\" class=\"sb-filtered-score\" style=\"margin:10px 0 4px\"></div>\n"
        "    <h3 class=\"analytics-sub-heading\">Missions</h3>\n"
        "    <div id=\"ana-missions-pies\"></div>\n"
        "    <h3 class=\"analytics-sub-heading\">Methods</h3>\n"
        "    <div id=\"ana-methods-pies\"></div>\n"
        "  </div>\n"
        "  <h2 class=\"all-results-heading\">All Results</h2>\n"
        "  <div id=\"all-results-wrapper\">\n"
        "    <div class=\"ar-filters\" id=\"ar-filters\"></div>\n"
        "    <button class=\"csv-download-btn\" onclick=\"downloadAllResultsCSV()\">&#11123; Download CSV</button>\n"
        "    <div id=\"all-results-table\"></div>\n"
        "  </div>\n"
        "  <script>\n"
        f"    const DATA = {data_js};\n"
        f"    const MISSION_META = {meta_js};\n"
        f"    const SCORES = {scores_js};\n"
        f"    const METHODS = {methods_js};\n"
    )
    tail = (
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    return head + js + tail


def _controls_html() -> str:
    return (
        "  <div class=\"controls\">\n"
        "    <div class=\"control-group\">\n"
        "      <label for=\"ctrl-dataset\">Dataset:</label>\n"
        "      <select id=\"ctrl-dataset\"></select>\n"
        "    </div>\n"
        "    <div class=\"control-group\">\n"
        "      <label for=\"ctrl-mission\">Mission:</label>\n"
        "      <select id=\"ctrl-mission\"></select>\n"
        "    </div>\n"
        "  </div>\n"
    )


