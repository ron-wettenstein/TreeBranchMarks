# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode
pip install -e .
pip install -e ".[dev]"       # include pytest

# Run all tests
pytest

# Run a single test file
pytest tests/test_params.py

# Run the example experiment (generates results/ and cache/)
python -m benchmarks.example_experiment
```

## Architecture

treebranchmarks is a parameter-sweep benchmarking framework. The execution hierarchy is:

```
Experiment → Mission[] → Task[] → Approach[]
```

**`Experiment`** (`core/experiment.py`) — top-level orchestrator. Runs missions, persists results to `results/{name}.json`, and generates the HTML report via `HtmlGenerator`. Supports `force_rerun`, `delete_dataset_cache`, `delete_model_cache`, `delete_results` flags.

**`Mission`** (`core/mission.py`) — one parameter sweep with a single free variable (n, m, or D). Iterates `n_values` in ascending order; if the linearly-extrapolated runtime for an approach exceeds `timeout_s` (default 600 s), the run is skipped and marked `is_estimated=True`. Dataset loading and model training are cached; mission results are not (that is the Experiment's job).

**`Task`** (`core/task.py`) — holds a list of `Approach` objects, runs them, returns a `TaskResult`. Handles warmup, repeated timing (`n_repeats`), and early stopping when the first repeat exceeds 10 s.

**`Approach`** (`core/approach.py`) — abstract base. Must implement `run()`, may implement `complexity_formula()` for calibrated runtime estimation. Key class attribute: `method = "shap" | "woodelf"` — this drives the scorer.

**`TreeParameters`** (`core/params.py`) — frozen dataclass with two-phase lifecycle: `T/D/L/F` set at train time via `partial_tree_params()`, then `n/m` filled in at run time via `with_run_params(n, m)`.

**`ModelWrapper`** (`core/model.py`) — trains a model and extracts `TreeParameters` from the trained artifact. `load_or_train()` caches the result under `cache/models/{dataset}/{config_md5}.joblib`. Implemented for XGBoost, LightGBM, and Random Forest in `models/`.

**`HtmlGenerator`** (`report/html_generator.py`) — produces a self-contained Plotly HTML file. All data is embedded as `const DATA` (JSON). Scoring is computed in Python (`_compute_scores()`) and embedded as `const SCORES`. The JS is all inline in the HTML template.

## Scoring

For each `(dataset, mission, task, n, m, D, ensemble)` group the two methods are compared:
- winner (lower time) → 100 pts; loser → `(winner / loser) × 100` pts
- Scores are averaged per mission and overall
- `is_estimated` rows **are** included in scoring (estimated times are valid performance data)
- Groups where either method's time ≤ 0 are skipped (division guard)

## Cache layout

```
cache/
  datasets/{name}/X.npy, y.npy
  models/{dataset}/{config_md5}.joblib
  calibration/{approach_name}.json
results/
  {experiment_name}.json    # full results
  {experiment_name}.html    # report
```

## Adding a new approach

Subclass `Approach`, set `method = "shap"` or `"woodelf"`, implement `run()`, optionally `complexity_formula()`. Pass via `extra_approaches` to a task factory (`BackgroundSHAPTask`, `PathDependentSHAPTask`).

## Adding a new task

Subclass `Approach` for each implementation, create a `Task(name=..., approaches=[...])` instance or a factory function following the pattern in `tasks/background_shap.py`.
