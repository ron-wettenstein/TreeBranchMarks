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
values.  Estimated times are shown with a trailing * and in italic.
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
                if ar.error:
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
                    "not_supported": ar.not_supported,
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
        if r.get("not_supported"):
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
        # Need at least one supported method with a competitor (supported or not_supported)
        if not supported or (len(supported) < 2 and not not_supported_methods):
            continue
        if supported:
            winner_time = min(supported.values())
            winner = min(supported, key=supported.get)
            scores = {m: (winner_time / t) * 100.0 for m, t in supported.items()}
            scores[winner] = 100.0
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

    def generate(self, result: ExperimentResult, output_path: Path) -> None:
        rows = _collect_rows(result)
        if not rows:
            raise ValueError("ExperimentResult contains no successful approach results.")

        data_js    = json.dumps(rows, separators=(",", ":"))
        meta_js    = json.dumps(_collect_mission_meta(result), separators=(",", ":"))
        scores_js  = json.dumps(_compute_scores(rows), separators=(",", ":"))
        methods_js = json.dumps(_collect_methods(result), separators=(",", ":"))
        html = _build_html(result.experiment_name, data_js, meta_js, scores_js, methods_js)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML assembly — use concatenation so JS braces need no escaping
# ---------------------------------------------------------------------------

def _build_html(experiment_name: str, data_js: str, meta_js: str, scores_js: str, methods_js: str) -> str:
    head = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>Treebranchmarks \u2014 {experiment_name}</title>\n"
        "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n"
        "  <style>\n" + _css() + "\n  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Treebranchmarks</h1>\n"
        f"  <div class=\"subtitle\">Experiment: <strong>{experiment_name}</strong></div>\n"
        "  <div id=\"scoreboard\"></div>\n"
        + _controls_html() +
        "  <details id=\"mission-details\" class=\"details-panel\">\n"
        "    <summary>Details: dataset, model &amp; approaches</summary>\n"
        "    <div id=\"mission-details-content\"></div>\n"
        "  </details>\n"
        "  <div id=\"mission-score-banner\"></div>\n"
        "  <div id=\"chart\"></div>\n"
        "  <div id=\"data-table\"></div>\n"
        "  <h2 class=\"all-results-heading\">All Results</h2>\n"
        "  <div id=\"all-results-wrapper\">\n"
        "    <div class=\"ar-filters\" id=\"ar-filters\"></div>\n"
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
    return head + _js() + tail


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


def _css() -> str:
    return """
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 20px 32px;
      background: #f0f2f5;
      color: #212529;
    }
    h1 { font-size: 1.5rem; color: #1f3a5f; margin-bottom: 0.2rem; }
    .subtitle { color: #6c757d; margin-bottom: 1.2rem; font-size: 0.9rem; }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 20px;
      background: white;
      border-radius: 8px;
      padding: 12px 18px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      margin-bottom: 14px;
      align-items: center;
    }
    .control-group { display: flex; align-items: center; gap: 6px; font-size: 0.88rem; }
    .control-group label { font-weight: 600; color: #495057; white-space: nowrap; }
    .control-group select {
      border: 1px solid #ced4da;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 0.87rem;
      background: #fff;
      cursor: pointer;
      min-width: 160px;
    }
    .control-group select:focus {
      outline: none;
      border-color: #1f77b4;
      box-shadow: 0 0 0 2px rgba(31,119,180,0.18);
    }

    #chart {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      margin-bottom: 14px;
      padding: 8px 8px 4px;
    }

    #data-table {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      padding: 16px 20px;
      overflow-x: auto;
    }
    #data-table p { color: #888; font-size: 0.9rem; }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 0.86rem;
    }
    thead th {
      background: #1f3a5f;
      color: white;
      padding: 8px 14px;
      text-align: left;
      font-weight: 600;
      white-space: nowrap;
    }
    thead th:first-child { border-radius: 4px 0 0 0; }
    thead th:last-child  { border-radius: 0 4px 0 0; }
    tbody tr:nth-child(even) { background: #f4f7fb; }
    tbody tr:hover { background: #e8f0fa; }
    tbody td {
      padding: 6px 14px;
      border-bottom: 1px solid #e2e8f0;
    }
    td.app-name { font-weight: 600; white-space: nowrap; color: #1f3a5f; }
    td.estimated { color: #888; font-style: italic; }
    td.missing { color: #bbb; }

    .details-panel {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      padding: 0;
      margin-bottom: 14px;
      overflow: hidden;
    }
    .details-panel summary {
      padding: 10px 18px;
      font-size: 0.88rem;
      font-weight: 600;
      color: #1f3a5f;
      cursor: pointer;
      user-select: none;
    }
    #mission-details-content {
      padding: 4px 20px 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px 24px;
      border-top: 1px solid #e9ecef;
    }
    .detail-card h3 {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #6c757d;
      margin: 12px 0 6px;
    }
    .detail-card dl {
      margin: 0;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 3px 10px;
      font-size: 0.84rem;
    }
    .detail-card dt { color: #6c757d; white-space: nowrap; }
    .detail-card dd { margin: 0; color: #212529; font-weight: 500; word-break: break-word; }
    .detail-card .col-list {
      font-size: 0.78rem;
      color: #495057;
      margin: 4px 0 0;
      line-height: 1.6;
    }
    .approach-row { margin-bottom: 6px; }
    .approach-row .aname { font-weight: 600; color: #1f3a5f; font-size: 0.84rem; }
    .approach-row .adesc { font-size: 0.79rem; color: #6c757d; margin-left: 4px; }

    .all-results-heading {
      font-size: 1.1rem;
      color: #1f3a5f;
      margin: 24px 0 8px;
    }
    #all-results-wrapper {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      padding: 16px 20px;
    }
    .ar-filters {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 32px;
      margin-bottom: 14px;
      align-items: flex-start;
      font-size: 0.85rem;
    }
    .ar-filter-group { display: flex; align-items: center; gap: 10px; }
    .ar-filter-group > label { font-weight: 600; color: #495057; white-space: nowrap; min-width: 16px; }
    .dual-range-wrap { display: flex; flex-direction: column; gap: 2px; min-width: 180px; }
    .dual-range-slider {
      position: relative;
      height: 22px;
    }
    .dual-range-slider input[type=range] {
      position: absolute;
      width: 100%;
      height: 4px;
      top: 9px;
      margin: 0; padding: 0;
      outline: none;
      -webkit-appearance: none;
      appearance: none;
      background: transparent;
      pointer-events: none;
    }
    .dual-range-slider input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 15px; height: 15px;
      border-radius: 50%;
      background: #1f77b4;
      border: 2px solid white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.28);
      cursor: pointer;
      pointer-events: all;
    }
    .dual-range-slider input[type=range]::-moz-range-thumb {
      width: 15px; height: 15px;
      border-radius: 50%;
      background: #1f77b4;
      border: 2px solid white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.28);
      cursor: pointer;
      pointer-events: all;
    }
    .drs-track {
      position: absolute;
      top: 9px; left: 0; right: 0;
      height: 4px;
      background: #dee2e6;
      border-radius: 2px;
      pointer-events: none;
    }
    .drs-fill {
      position: absolute;
      top: 9px;
      height: 4px;
      background: #1f77b4;
      border-radius: 2px;
      pointer-events: none;
    }
    .drs-vals {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: #1f77b4;
      font-weight: 600;
      padding: 0 1px;
    }
    #all-results-table { overflow-x: auto; }
    #all-results-table table { font-size: 0.83rem; }
    #all-results-table td { color: #333; }
    #all-results-table td.time-cell { font-variant-numeric: tabular-nums; }

    /* Scoreboard */
    #scoreboard {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      padding: 14px 20px 12px;
      margin-bottom: 14px;
      display: flex;
      gap: 28px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .sb-left  { flex: 0 0 auto; }
    .sb-divider { width: 1px; background: #dee2e6; align-self: stretch; margin: 4px 0; }
    .sb-right { flex: 1; min-width: 240px; }
    .sb-sliders { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 12px; }
    .sb-filtered-score { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; min-height: 28px; }
    .scoreboard-title {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #6c757d;
      margin-bottom: 10px;
    }
    .score-table {
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.87rem;
      width: auto;
    }
    .score-table thead th {
      background: none;
      color: #6c757d;
      font-size: 0.77rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0 20px 6px 4px;
      border-bottom: 1px solid #dee2e6;
      white-space: nowrap;
    }
    .score-table tbody tr:hover { background: #f8f9fa; }
    .score-table tbody td {
      padding: 7px 20px 7px 4px;
      border-bottom: none;
    }
    .score-table .team-name {
      font-weight: 700;
      font-size: 0.9rem;
      white-space: nowrap;
      padding-right: 28px;
    }
    .woodelf-color { color: #2ca02c; }
    .shap-color    { color: #1f77b4; }
    .score-cell {
      font-variant-numeric: tabular-nums;
      font-size: 1.05rem;
      font-weight: 700;
    }
    .score-cell.winner { font-size: 1.15rem; }
    .score-bar-wrap { display: flex; align-items: center; gap: 8px; }
    .score-bar {
      height: 6px;
      border-radius: 3px;
      min-width: 4px;
      max-width: 120px;
    }
    .score-bar   { background: #2ca02c; } /* default; overridden inline per method */

    /* Mission score banner */
    #mission-score-banner {
      display: flex;
      gap: 10px;
      align-items: center;
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.09);
      padding: 10px 20px;
      margin-bottom: 14px;
      font-size: 0.86rem;
    }
    #mission-score-banner:empty { display: none; }
    .msb-label {
      color: #6c757d;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      margin-right: 4px;
    }
    .msb-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 10px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 0.88rem;
    }
    .msb-badge.woodelf { background: #eafaea; color: #1a7a1a; }
    .msb-badge.shap    { background: #e8f2fc; color: #1056a0; }
    .msb-badge.winner  { box-shadow: 0 0 0 2px currentColor; }
    .msb-sep { color: #ced4da; font-size: 1rem; }
    .filter-row { display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; font-size: 0.82rem; }
    .fr-label { font-weight: 600; color: #495057; white-space: nowrap; padding-top: 3px; }
    .filter-chips { display: flex; flex-wrap: wrap; gap: 5px; }
    .filter-chip { display: inline-flex; align-items: center; gap: 4px; font-size: 0.79rem; padding: 2px 9px; border-radius: 12px; border: 1px solid #ced4da; cursor: pointer; color: #495057; background: white; }
    .filter-chip input[type=checkbox] { margin: 0; cursor: pointer; accent-color: #1f77b4; width: 12px; height: 12px; }
    .remove-fast-wrap { display: inline-flex; align-items: center; gap: 5px; font-size: 0.82rem; color: #495057; cursor: pointer; }
    .remove-fast-wrap input[type=checkbox] { cursor: pointer; accent-color: #e67e22; margin: 0; }
"""


def _js() -> str:
    # Plain string — no Python f-string, so { and } are literal JS.
    return r"""
    // -----------------------------------------------------------------------
    // Method metadata — keyed by method.name, built from METHODS constant.
    // -----------------------------------------------------------------------
    var _methodMap = {};
    METHODS.forEach(function(m) { _methodMap[m.name] = m; });
    function methodLabel(name) {
      return (_methodMap[name] && _methodMap[name].label) || name || 'unknown';
    }
    var _scoreColors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd','#8c564b','#e377c2'];
    function methodColor(name) {
      var idx = (SCORES.methods || []).indexOf(name);
      return _scoreColors[idx >= 0 ? idx % _scoreColors.length : 0];
    }
    function renderMethodBadge(m, val, maxScore) {
      var win = val >= maxScore - 0.001;
      var color = methodColor(m);
      var ns = val === 0 && !win;
      var label = methodLabel(m) + ': ' + (ns ? 'N/A' : val.toFixed(1) + (win ? ' \u2605' : ''));
      return '<span class="msb-badge" style="background:' + color +
             ';color:#fff;opacity:' + (win ? '1' : '0.7') + ';border:1px solid ' + color + '">' +
             label + '</span>';
    }

    // Scoring — pre-computed by Python, embedded as SCORES constant.
    // SCORES = { methods: [...], overall: { scores: {method: avg}, n }, by_mission: {...} }
    // -----------------------------------------------------------------------

    function renderScoreboard() {
      var el = document.getElementById('scoreboard');
      if (!SCORES.overall) { el.style.display = 'none'; return; }

      function scoreCell(scoresObj, methodName) {
        if (!scoresObj || !scoresObj.scores) return '<td class="score-cell" style="color:#bbb">\u2014</td>';
        var val = scoresObj.scores[methodName];
        if (val === undefined) return '<td class="score-cell" style="color:#bbb">\u2014</td>';
        var maxVal = Math.max.apply(null, Object.keys(scoresObj.scores).map(function(m){ return scoresObj.scores[m]; }));
        var isWin = val >= maxVal - 0.001;
        var isNs = val === 0 && !isWin;
        var color = methodColor(methodName);
        return '<td class="score-cell' + (isWin ? ' winner' : '') + '">' +
               '<span style="color:' + color + (isNs ? ';opacity:0.5' : '') + '">' +
               (isNs ? 'N/A' : val.toFixed(1) + (isWin ? ' \u2605' : '')) + '</span>' +
               '</td>';
      }

      // --- Left: overall summary table ---
      var methods = SCORES.methods || [];
      var leftTitle = methods.map(function(m){ return methodLabel(m); }).join(' vs ');
      var leftHtml = '<div class="scoreboard-title">Score Summary &mdash; ' + leftTitle + '</div>';
      leftHtml += '<table class="score-table"><thead><tr>';
      leftHtml += '<th></th><th>Overall (' + SCORES.overall.n + ' runs)</th>';
      leftHtml += '</tr></thead><tbody>';
      methods.forEach(function(m) {
        leftHtml += '<tr><td class="team-name" style="color:' + methodColor(m) + '">' + methodLabel(m) + '</td>';
        leftHtml += scoreCell(SCORES.overall, m);
        leftHtml += '</tr>';
      });
      leftHtml += '</tbody></table>';

      // --- Right: filter sliders + live filtered score ---
      var SB_PARAMS = ['n', 'm', 'D'];
      var slidersHtml = '<div class="scoreboard-title">Filtered Score</div><div class="sb-sliders">';
      SB_PARAMS.forEach(function(p) {
        var vals = getUnique(p);
        var maxI = vals.length - 1;
        slidersHtml +=
          '<div class="ar-filter-group">' +
          '<label>' + p + ':</label>' +
          '<div class="dual-range-wrap">' +
            '<div class="dual-range-slider">' +
              '<div class="drs-track"></div>' +
              '<div class="drs-fill" id="sb-drs-' + p + '-fill"></div>' +
              '<input type="range" id="sb-drs-' + p + '-lo" min="0" max="' + maxI + '" value="0">' +
              '<input type="range" id="sb-drs-' + p + '-hi" min="0" max="' + maxI + '" value="' + maxI + '">' +
            '</div>' +
            '<div class="drs-vals">' +
              '<span id="sb-drs-' + p + '-lo-val">' + vals[0] + '</span>' +
              '<span id="sb-drs-' + p + '-hi-val">' + vals[maxI] + '</span>' +
            '</div>' +
          '</div></div>';
      });
      slidersHtml += '</div>';
      var sbTaskVals = getUnique('task');
      slidersHtml += '<div class="filter-row"><span class="fr-label">Tasks:</span><div class="filter-chips">';
      sbTaskVals.forEach(function(t) {
        slidersHtml += '<label class="filter-chip"><input type="checkbox" class="sb-task-cb" value="' + t + '" checked> ' + taskDisplayName(t) + '</label>';
      });
      slidersHtml += '</div></div>';
      slidersHtml += '<div class="filter-row"><label class="remove-fast-wrap"><input type="checkbox" id="sb-remove-fast"> Runtime &gt; 10s</label></div>';
      slidersHtml += '<div id="sb-filtered-score" class="sb-filtered-score"></div>';

      el.innerHTML =
        '<div class="sb-left">' + leftHtml + '</div>' +
        '<div class="sb-divider"></div>' +
        '<div class="sb-right">' + slidersHtml + '</div>';

      // --- Wire up slider logic ---
      var sbVals = {}, sbState = {};
      SB_PARAMS.forEach(function(p) {
        sbVals[p]  = getUnique(p);
        sbState[p] = { lo: 0, hi: sbVals[p].length - 1 };
      });

      function updateSbFill(p) {
        var n = sbVals[p].length;
        var fill = document.getElementById('sb-drs-' + p + '-fill');
        if (!fill || n <= 1) return;
        var pct = 100 / (n - 1);
        fill.style.left  = (sbState[p].lo * pct) + '%';
        fill.style.width = ((sbState[p].hi - sbState[p].lo) * pct) + '%';
      }

      function computeScoreFromRows(rows, removeFast) {
        var groups = {};
        rows.forEach(function(r) {
          if (!r.method) return;
          var key = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.ensemble].join('||');
          if (!groups[key]) groups[key] = { times: {}, notSupported: {} };
          if (r.not_supported) {
            groups[key].notSupported[r.method] = true;
          } else {
            if (!groups[key].times[r.method]) groups[key].times[r.method] = [];
            groups[key].times[r.method].push(r.running_time);
          }
        });
        var totals = {}, counts = {}, nGroups = 0;
        Object.keys(groups).forEach(function(key) {
          var g = groups[key];
          var times = {};
          Object.keys(g.times).forEach(function(m) {
            var avg = g.times[m].reduce(function(a,b){return a+b;},0) / g.times[m].length;
            if (avg > 0) times[m] = avg;
          });
          var nsKeys = Object.keys(g.notSupported);
          // Need at least one supported method with a competitor
          if (Object.keys(times).length === 0) return;
          if (Object.keys(times).length < 2 && nsKeys.length === 0) return;
          var minT = Math.min.apply(null, Object.keys(times).map(function(m){ return times[m]; }));
          if (removeFast) {
            var allFast = Object.keys(times).every(function(m){ return times[m] < 10; });
            if (allFast) return;
          }
          var winner = Object.keys(times).filter(function(m){ return times[m] <= minT; })[0];
          Object.keys(times).forEach(function(m) {
            var score = m === winner ? 100 : (minT / times[m]) * 100;
            totals[m] = (totals[m] || 0) + score;
            counts[m] = (counts[m] || 0) + 1;
          });
          nsKeys.forEach(function(m) {
            totals[m] = (totals[m] || 0) + 0;
            counts[m] = (counts[m] || 0) + 1;
          });
          nGroups++;
        });
        if (!nGroups) return null;
        var scores = {};
        Object.keys(totals).forEach(function(m) { scores[m] = totals[m] / counts[m]; });
        return { scores: scores, n: nGroups };
      }

      function updateFilteredScore() {
        var nLo = sbVals.n[sbState.n.lo], nHi = sbVals.n[sbState.n.hi];
        var mLo = sbVals.m[sbState.m.lo], mHi = sbVals.m[sbState.m.hi];
        var dLo = sbVals.D[sbState.D.lo], dHi = sbVals.D[sbState.D.hi];
        var selTasks = [];
        document.querySelectorAll('.sb-task-cb').forEach(function(cb) {
          if (cb.checked) selTasks.push(cb.value);
        });
        var rmFastEl = document.getElementById('sb-remove-fast');
        var removeFast = rmFastEl ? rmFastEl.checked : false;
        var filtered = DATA.filter(function(r) {
          return r.n >= nLo && r.n <= nHi && r.m >= mLo && r.m <= mHi && r.D >= dLo && r.D <= dHi &&
                 (selTasks.length === 0 || selTasks.indexOf(r.task) !== -1);
        });
        var result = computeScoreFromRows(filtered, removeFast);
        var el2 = document.getElementById('sb-filtered-score');
        if (!result) {
          el2.innerHTML = '<span style="color:#adb5bd;font-size:0.83rem">No comparable runs in range.</span>';
          return;
        }
        var maxScore = Math.max.apply(null, Object.keys(result.scores).map(function(m){ return result.scores[m]; }));
        var html = Object.keys(result.scores).sort().map(function(m) {
          return renderMethodBadge(m, result.scores[m], maxScore);
        }).join('<span class="msb-sep">vs</span>');
        html += '<span style="color:#adb5bd;font-size:0.78rem;margin-left:6px">(' + result.n + ' runs)</span>';
        el2.innerHTML = html;
      }

      SB_PARAMS.forEach(function(p) {
        var loEl = document.getElementById('sb-drs-' + p + '-lo');
        var hiEl = document.getElementById('sb-drs-' + p + '-hi');
        loEl.addEventListener('input', function() {
          if (parseInt(loEl.value) > parseInt(hiEl.value)) loEl.value = hiEl.value;
          sbState[p].lo = parseInt(loEl.value);
          sbState[p].hi = parseInt(hiEl.value);
          document.getElementById('sb-drs-' + p + '-lo-val').textContent = sbVals[p][sbState[p].lo];
          document.getElementById('sb-drs-' + p + '-hi-val').textContent = sbVals[p][sbState[p].hi];
          updateSbFill(p);
          updateFilteredScore();
        });
        hiEl.addEventListener('input', function() {
          if (parseInt(hiEl.value) < parseInt(loEl.value)) hiEl.value = loEl.value;
          sbState[p].lo = parseInt(loEl.value);
          sbState[p].hi = parseInt(hiEl.value);
          document.getElementById('sb-drs-' + p + '-lo-val').textContent = sbVals[p][sbState[p].lo];
          document.getElementById('sb-drs-' + p + '-hi-val').textContent = sbVals[p][sbState[p].hi];
          updateSbFill(p);
          updateFilteredScore();
        });
        updateSbFill(p);
      });

      document.querySelectorAll('.sb-task-cb').forEach(function(cb) {
        cb.addEventListener('change', updateFilteredScore);
      });
      var sbRmFastEl = document.getElementById('sb-remove-fast');
      if (sbRmFastEl) sbRmFastEl.addEventListener('change', updateFilteredScore);

      updateFilteredScore();
    }

    function renderMissionScore(missionName) {
      var el = document.getElementById('mission-score-banner');
      var entry = (SCORES.by_mission || {})[missionName];
      if (!entry || !entry.scores) { el.innerHTML = ''; return; }

      var maxScore = Math.max.apply(null, Object.keys(entry.scores).map(function(m){ return entry.scores[m]; }));
      var badges = Object.keys(entry.scores).sort().map(function(m) {
        return renderMethodBadge(m, entry.scores[m], maxScore);
      });

      el.innerHTML =
        '<span class="msb-label">Mission Score:</span>' +
        badges.join('<span class="msb-sep">vs</span>') +
        '<span style="color:#adb5bd;font-size:0.78rem;margin-left:6px">avg of ' +
        entry.n + ' run' + (entry.n > 1 ? 's' : '') + '</span>';
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    function taskDisplayName(t) {
      if (t === 'background_shap')    return 'Background SHAP';
      if (t === 'path_dependent_shap') return 'Path-Dependent SHAP';
      return t;
    }

    function unique(arr) {
      return [...new Set(arr)].sort(function(a, b) {
        return typeof a === 'number' ? a - b : String(a).localeCompare(String(b));
      });
    }

    function getUnique(key, subset) {
      var src = subset !== undefined ? subset : DATA;
      return unique(src.map(function(r) { return r[key]; }));
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var allDatasets = getUnique('dataset');
    var state = {
      dataset: allDatasets[0],
      mission: null,
    };

    // -----------------------------------------------------------------------
    // Cascading options helpers
    // -----------------------------------------------------------------------
    function rowsForDataset() {
      return DATA.filter(function(r) { return r.dataset === state.dataset; });
    }

    function rowsForMission() {
      return rowsForDataset().filter(function(r) { return r.mission === state.mission; });
    }

    function missionsForDataset() { return getUnique('mission', rowsForDataset()); }

    // -----------------------------------------------------------------------
    // Auto-detect x-axis from the current mission's data
    // -----------------------------------------------------------------------
    var X_LABELS = {
      n: 'n \u2014 explain set size',
      m: 'm \u2014 background size',
      D: 'D \u2014 max tree depth',
    };

    function detectXParam(rows) {
      var ns = getUnique('n', rows);
      var ms = getUnique('m', rows);
      var ds = getUnique('D', rows);
      if (ns.length > 1) return 'n';
      if (ds.length > 1) return 'D';
      if (ms.length > 1) return 'm';
      return 'n';
    }

    // -----------------------------------------------------------------------
    // Chart
    // -----------------------------------------------------------------------
    function buildTraces(rows, xp) {
      var approaches = unique(rows.map(function(r) { return r.approach; }));
      return approaches.map(function(app) {
        var appRows = rows
          .filter(function(r) { return r.approach === app && !r.not_supported; })
          .sort(function(a, b) { return a[xp] - b[xp]; });
        if (!appRows.length) return null;  // all points unsupported — skip trace

        var label = methodLabel(appRows[0].method);
        return {
          x: appRows.map(function(r) { return r[xp]; }),
          y: appRows.map(function(r) { return r.running_time; }),
          error_y: {
            type: 'data',
            array: appRows.map(function(r) { return r.is_estimated ? 0 : r.std_s; }),
            visible: true,
          },
          mode: 'lines+markers',
          name: label,
          marker: {
            symbol: appRows.map(function(r) { return r.is_estimated ? 'circle-open' : 'circle'; }),
            size: 9,
          },
          customdata: appRows.map(function(r) { return r.is_estimated ? '\u2009\u2605\u202festimated' : ''; }),
          hovertemplate:
            '<b>' + label + '</b><br>' +
            xp + ': %{x}<br>' +
            'time: %{y:.4f} s%{customdata}' +
            '<extra></extra>',
        };
      });
    }

    function shouldUseLogScale(vals) {
      var nums = vals.filter(function(v) { return v > 0; });
      if (nums.length < 2) return false;
      var mn = Math.min.apply(null, nums);
      var mx = Math.max.apply(null, nums);
      return mx / mn >= 20;
    }

    function renderChart(rows, xp) {
      var traces = buildTraces(rows, xp).filter(function(t) { return t !== null; });
      var xVals = getUnique(xp, rows);
      var supportedRows = rows.filter(function(r) { return !r.not_supported; });
      var yVals = supportedRows.map(function(r) { return r.running_time; }).filter(function(v) { return v > 0; });
      var useLogX = shouldUseLogScale(xVals);
      var useLogY = shouldUseLogScale(yVals);
      var layout = {
        xaxis: { title: X_LABELS[xp] || xp, type: useLogX ? 'log' : 'linear' },
        yaxis: { title: 'Time (s)', type: useLogY ? 'log' : 'linear' },
        legend: { title: { text: 'Method\u2003\u25cb\u202f=\u202festimated' } },
        template: 'plotly_white',
        margin: { t: 30, r: 20, l: 70, b: 55 },
        height: 460,
      };
      Plotly.react('chart', traces, layout);
    }

    // -----------------------------------------------------------------------
    // Table
    // -----------------------------------------------------------------------
    function renderTable(rows, xp) {
      var el = document.getElementById('data-table');
      if (rows.length === 0) {
        el.innerHTML = '<p>No data matches the current selection.</p>';
        return;
      }

      var xVals     = getUnique(xp, rows);
      var approaches = getUnique('approach', rows);

      // index[approach][xVal] = {t, est}
      var index = {};
      rows.forEach(function(r) {
        if (!index[r.approach]) index[r.approach] = {};
        index[r.approach][r[xp]] = { t: r.running_time, est: r.is_estimated };
      });

      var html = '<table><thead><tr><th>Approach</th>';
      xVals.forEach(function(v) { html += '<th>' + xp + '\u202f=\u202f' + v + '</th>'; });
      html += '</tr></thead><tbody>';

      approaches.forEach(function(app) {
        html += '<tr><td class="app-name">' + app + '</td>';
        xVals.forEach(function(v) {
          var cell = (index[app] || {})[v];
          if (cell !== undefined) {
            var val = cell.t.toFixed(3);
            html += '<td class="' + (cell.est ? 'estimated' : '') + '">'
                  + val + (cell.est ? '*' : '') + '</td>';
          } else {
            html += '<td class="missing">\u2014</td>';
          }
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      el.innerHTML = html;
    }

    // -----------------------------------------------------------------------
    // Details panel
    // -----------------------------------------------------------------------
    function renderDetails(missionName, rows) {
      var el = document.getElementById('mission-details-content');
      var m = MISSION_META[missionName];
      if (!m) { el.innerHTML = '<p style="color:#888;font-size:0.85rem">No metadata available.</p>'; return; }

      var html = '';

      // --- Dataset card ---
      var ds = m.dataset || {};
      html += '<div class="detail-card">';
      html += '<h3>Dataset</h3>';
      html += '<dl>';
      html += '<dt>Name</dt><dd>' + (ds.name || '\u2014') + '</dd>';
      html += '<dt>Rows</dt><dd>' + (ds.n_samples != null ? ds.n_samples.toLocaleString() : '\u2014') + '</dd>';
      html += '<dt>Features</dt><dd>' + (ds.n_features != null ? ds.n_features : '\u2014') + '</dd>';
      if (ds.columns && ds.columns.length) {
        var shown = ds.columns.slice(0, 10);
        var extra = ds.columns.length - shown.length;
        html += '<dt>Feature names</dt><dd>' + shown.join(', ') + (extra > 0 ? ' \u2026 +' + extra + ' more' : '') + '</dd>';
      }
      html += '</dl>';
      html += '</div>';

      // --- Model(s) card ---
      var models = m.models || [];
      html += '<div class="detail-card">';
      html += '<h3>Model' + (models.length > 1 ? 's' : '') + '</h3>';
      models.forEach(function(mod) {
        html += '<dl>';
        html += '<dt>Type</dt><dd>' + (mod.ensemble_type || '\u2014') + '</dd>';
        var hp = mod.hyperparams || {};
        Object.keys(hp).forEach(function(k) {
          html += '<dt>' + k + '</dt><dd>' + hp[k] + '</dd>';
        });
        // Pull T and L from the first data row for this mission
        var sampleRow = (rows || []).find(function(r) {
          return r.ensemble === mod.ensemble_type;
        });
        if (sampleRow) {
          var totalLeaves = Math.round(sampleRow.T * sampleRow.L);
          html += '<dt>trees (T)</dt><dd>' + sampleRow.T + '</dd>';
          html += '<dt>avg leaves/tree (L)</dt><dd>' + sampleRow.L.toFixed(1) + '</dd>';
          html += '<dt>total leaves</dt><dd>' + totalLeaves + '</dd>';
        }
        html += '</dl>';
      });
      html += '</div>';

      // --- Task & approaches card ---
      var tasks = m.tasks || [];
      html += '<div class="detail-card">';
      html += '<h3>Task &amp; Approaches</h3>';
      tasks.forEach(function(t) {
        html += '<p style="font-size:0.85rem;font-weight:600;color:#1f3a5f;margin:6px 0 4px">' + t.name + '</p>';
        (t.approaches || []).forEach(function(a) {
          html += '<div class="approach-row">'
                + '<span class="aname">' + a.name + '</span>'
                + (a.description ? '<span class="adesc">\u2014 ' + a.description + '</span>' : '')
                + '</div>';
        });
      });
      html += '</div>';

      // --- Sweep card ---
      html += '<div class="detail-card">';
      html += '<h3>Mission Sweep</h3>';
      html += '<dl>';
      html += '<dt>n values</dt><dd>' + (m.n_values || []).join(', ') + '</dd>';
      html += '<dt>m values</dt><dd>' + (m.m_values || []).join(', ') + '</dd>';
      html += '</dl>';
      html += '</div>';

      el.innerHTML = html;
    }

    // -----------------------------------------------------------------------
    // Main render
    // -----------------------------------------------------------------------
    function render() {
      var rows = rowsForMission();

      renderDetails(state.mission, rows);
      renderMissionScore(state.mission);

      if (rows.length === 0) {
        Plotly.react('chart', [], { title: 'No data for this selection', template: 'plotly_white' });
        document.getElementById('data-table').innerHTML = '<p>No data matches the current selection.</p>';
        return;
      }

      var xp = detectXParam(rows);
      renderChart(rows, xp);
      renderTable(rows, xp);
    }

    // -----------------------------------------------------------------------
    // Select population helpers
    // -----------------------------------------------------------------------
    function populateSelect(id, values, current) {
      var el = document.getElementById(id);
      el.innerHTML = '';
      values.forEach(function(v) {
        var opt = document.createElement('option');
        opt.value = v;
        opt.text  = String(v);
        if (String(v) === String(current)) opt.selected = true;
        el.appendChild(opt);
      });
      // Return the actually-selected value (first if current wasn't in list).
      return el.options[el.selectedIndex] ? el.options[el.selectedIndex].value : (values[0] || null);
    }

    function refreshMissionSelect() {
      var missions = missionsForDataset();
      if (missions.indexOf(state.mission) === -1) state.mission = missions[0] || null;
      populateSelect('ctrl-mission', missions, state.mission);
      state.mission = missions.indexOf(state.mission) !== -1 ? state.mission : (missions[0] || null);
    }

    // -----------------------------------------------------------------------
    // Initialise
    // -----------------------------------------------------------------------
    populateSelect('ctrl-dataset', allDatasets, state.dataset);
    refreshMissionSelect();

    document.getElementById('ctrl-dataset').addEventListener('change', function(e) {
      state.dataset = e.target.value;
      refreshMissionSelect();
      render();
    });
    document.getElementById('ctrl-mission').addEventListener('change', function(e) {
      state.mission = e.target.value;
      render();
    });

    renderScoreboard();
    render();

    // -----------------------------------------------------------------------
    // All-results table — dual-range slider filters
    // -----------------------------------------------------------------------
    var AR_COLS    = ['dataset', 'approach', 'task', 'mission', 'n', 'm', 'D', 'T', 'L', 'F', 'ensemble', 'running_time', 'is_estimated'];
    var AR_HEADERS = ['Dataset', 'Approach', 'Task', 'Mission', 'n', 'm', 'D', 'T', 'L', 'F', 'Ensemble', 'Time (s)', 'Est?'];
    var AR_PARAMS  = ['n', 'm', 'D'];

    var arVals  = {};   // param -> sorted unique value array
    var arState = {};   // param -> {lo: idx, hi: idx}

    function buildDualSliders() {
      AR_PARAMS.forEach(function(p) {
        arVals[p]  = getUnique(p);
        arState[p] = { lo: 0, hi: arVals[p].length - 1 };
      });

      var html = '';
      AR_PARAMS.forEach(function(p) {
        var vals = arVals[p];
        var maxI = vals.length - 1;
        html +=
          '<div class="ar-filter-group">' +
          '<label>' + p + ':</label>' +
          '<div class="dual-range-wrap">' +
            '<div class="dual-range-slider">' +
              '<div class="drs-track"></div>' +
              '<div class="drs-fill" id="drs-' + p + '-fill"></div>' +
              '<input type="range" id="drs-' + p + '-lo" min="0" max="' + maxI + '" value="0">' +
              '<input type="range" id="drs-' + p + '-hi" min="0" max="' + maxI + '" value="' + maxI + '">' +
            '</div>' +
            '<div class="drs-vals">' +
              '<span id="drs-' + p + '-lo-val">' + vals[0] + '</span>' +
              '<span id="drs-' + p + '-hi-val">' + vals[maxI] + '</span>' +
            '</div>' +
          '</div>' +
          '</div>';
      });
      var arTaskVals = getUnique('task');
      html += '<div class="ar-filter-group"><span class="fr-label">Tasks:</span><div class="filter-chips">';
      arTaskVals.forEach(function(t) {
        html += '<label class="filter-chip"><input type="checkbox" class="ar-task-cb" value="' + t + '" checked> ' + taskDisplayName(t) + '</label>';
      });
      html += '</div></div>';
      html += '<div class="ar-filter-group"><label class="remove-fast-wrap"><input type="checkbox" id="ar-remove-fast"> Runtime &gt; 10s</label></div>';
      document.getElementById('ar-filters').innerHTML = html;

      document.querySelectorAll('.ar-task-cb').forEach(function(cb) {
        cb.addEventListener('change', renderAllResultsTable);
      });
      var arRmFastEl = document.getElementById('ar-remove-fast');
      if (arRmFastEl) arRmFastEl.addEventListener('change', renderAllResultsTable);

      AR_PARAMS.forEach(function(p) {
        var loEl = document.getElementById('drs-' + p + '-lo');
        var hiEl = document.getElementById('drs-' + p + '-hi');
        loEl.addEventListener('input', function() {
          if (parseInt(loEl.value) > parseInt(hiEl.value)) loEl.value = hiEl.value;
          onSliderChange(p);
        });
        hiEl.addEventListener('input', function() {
          if (parseInt(hiEl.value) < parseInt(loEl.value)) hiEl.value = loEl.value;
          onSliderChange(p);
        });
        updateFill(p);
      });
    }

    function onSliderChange(p) {
      var lo = parseInt(document.getElementById('drs-' + p + '-lo').value);
      var hi = parseInt(document.getElementById('drs-' + p + '-hi').value);
      arState[p] = { lo: lo, hi: hi };
      document.getElementById('drs-' + p + '-lo-val').textContent = arVals[p][lo];
      document.getElementById('drs-' + p + '-hi-val').textContent = arVals[p][hi];
      updateFill(p);
      renderAllResultsTable();
    }

    function updateFill(p) {
      var n = arVals[p].length;
      var fill = document.getElementById('drs-' + p + '-fill');
      if (!fill) return;
      if (n <= 1) { fill.style.left = '0%'; fill.style.width = '100%'; return; }
      var pct = 100 / (n - 1);
      fill.style.left  = (arState[p].lo * pct) + '%';
      fill.style.width = ((arState[p].hi - arState[p].lo) * pct) + '%';
    }

    function renderAllResultsTable() {
      var selArTasks = [];
      document.querySelectorAll('.ar-task-cb').forEach(function(cb) {
        if (cb.checked) selArTasks.push(cb.value);
      });
      var arRmFast = document.getElementById('ar-remove-fast') ? document.getElementById('ar-remove-fast').checked : false;
      var fastKeys = null;
      if (arRmFast) {
        var fg = {};
        DATA.forEach(function(r) {
          if (r.method !== 'woodelf' && r.method !== 'shap') return;
          var k = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.ensemble].join('||');
          if (!fg[k]) fg[k] = {};
          if (fg[k][r.method] === undefined || r.running_time < fg[k][r.method]) fg[k][r.method] = r.running_time;
        });
        fastKeys = {};
        Object.keys(fg).forEach(function(k) {
          var g = fg[k];
          if (g.woodelf !== undefined && g.shap !== undefined && g.woodelf < 10 && g.shap < 10) fastKeys[k] = true;
        });
      }
      var filtered = DATA.filter(function(r) {
        if (!AR_PARAMS.every(function(p) {
          return r[p] >= arVals[p][arState[p].lo] && r[p] <= arVals[p][arState[p].hi];
        })) return false;
        if (selArTasks.length > 0 && selArTasks.indexOf(r.task) === -1) return false;
        if (fastKeys) {
          var k = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.ensemble].join('||');
          if (fastKeys[k]) return false;
        }
        return true;
      });

      var html = '<table><thead><tr>';
      AR_HEADERS.forEach(function(h) { html += '<th>' + h + '</th>'; });
      html += '</tr></thead><tbody>';
      filtered.forEach(function(r) {
        html += '<tr>';
        AR_COLS.forEach(function(c) {
          if (c === 'running_time') {
            html += '<td class="time-cell' + (r.is_estimated ? ' estimated' : '') + '">'
                  + Number(r[c]).toFixed(4) + (r.is_estimated ? '*' : '') + '</td>';
          } else if (c === 'is_estimated') {
            html += '<td>' + (r[c] ? '\u2605' : '') + '</td>';
          } else {
            html += '<td>' + r[c] + '</td>';
          }
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      document.getElementById('all-results-table').innerHTML = html;
    }

    buildDualSliders();
    renderAllResultsTable();
"""
