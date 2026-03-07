# treebranchmarks

A benchmarking framework for comparing SHAP algorithms on decision tree ensembles.

treebranchmarks measures and visualises how the runtime of different SHAP implementations scales across the six parameters that govern their complexity:

| Parameter | Meaning |
|---|---|
| `n` | rows being explained |
| `m` | rows in the background dataset |
| `T` | number of trees |
| `D` | tree depth |
| `L` | average leaves per tree |
| `F` | number of features |

Results are summarised in a self-contained, interactive HTML report with log-scale line charts, a sortable all-results table, and a head-to-head scoreboard comparing **SHAP** vs **Woodelf**.

---

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[kaggle]"   # Kaggle dataset support
pip install -e ".[openml]"   # OpenML dataset support
pip install -e ".[dev]"      # pytest
```

Python 3.10+ required.

---

## Quick start

```python
from pathlib import Path
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import CaliforniaHousingDataset
from treebranchmarks.models import LightGBMWrapper
from treebranchmarks.tasks import PathDependentSHAPTask, BackgroundSHAPTask

dataset = CaliforniaHousingDataset()
model   = {ModelConfig(EnsembleType.LIGHTGBM, {"n_estimators": 100, "max_depth": 6}): LightGBMWrapper()}

experiment = Experiment(
    name="my_benchmark",
    missions=[
        # Sweep n — how does runtime grow with the number of rows to explain?
        Mission(MissionConfig(
            name="path-dep: sweep n",
            dataset=dataset,
            model_wrappers=model,
            tasks=[PathDependentSHAPTask()],
            n_values=[10, 100, 1_000, 10_000],
            m_values=[0],
        )),
        # Sweep m — how does runtime grow with the background size?
        Mission(MissionConfig(
            name="background: sweep m",
            dataset=dataset,
            model_wrappers=model,
            tasks=[BackgroundSHAPTask()],
            n_values=[200],
            m_values=[1, 10, 100, 1_000, 10_000],
        )),
    ],
)

experiment.run()
report = experiment.generate_html()
print(f"Report: {report.resolve()}")
```

See [`benchmarks/example_experiment.py`](benchmarks/example_experiment.py) for a fuller example.

---

## How it works

### Key concepts

```
Experiment
└── Mission[]          — one free parameter swept (n, m, or D)
    └── Task[]         — one algorithmic problem (e.g. Background SHAP)
        └── Approach[] — one concrete implementation (SHAP / Woodelf)
```

**Mission** — a single parameter sweep. Each mission varies exactly one dimension (e.g. `n_values=[1, 10, 100, 1000]`) while keeping everything else fixed. Missions are run smallest-to-largest; if the linearly-extrapolated runtime for an approach exceeds the timeout (default 10 min), that point is skipped and marked as estimated.

**Task** — groups one or more Approach objects under a shared problem definition. Built-in tasks:

| Task | Complexity |
|---|---|
| `PathDependentSHAPTask` | O(T · L² · n) |
| `BackgroundSHAPTask` | O(T · L · n · m) |

**Approach** — one implementation. Each approach has a `method` tag (`"shap"` or `"woodelf"`) used by the scorer.

### Timeout and estimation

Missions iterate `n_values` in ascending order. After each successfully measured run the time is stored. Before the next (larger) `n`, the runtime is linearly extrapolated. If the extrapolated time exceeds `timeout_s`, the run is skipped and its result is flagged `is_estimated=True`. Estimated times appear as open circles in the chart.

### Scoring

For each `(dataset, mission, task, n, m, D, ensemble)` group the two methods are compared:

- **Winner** (lower time) → 100 points
- **Loser** → `(winner_time / loser_time) × 100` points

The scoreboard shows the average score across all groups, and can be filtered interactively by `n`, `m`, and `D` range.

---

## Project structure

```
treebranchmarks/
├── core/
│   ├── params.py        # TreeParameters dataclass, EnsembleType enum
│   ├── dataset.py       # Dataset ABC
│   ├── model.py         # ModelWrapper ABC, ModelConfig, TrainedModel
│   ├── approach.py      # Approach ABC, ApproachOutput
│   ├── task.py          # Task, TaskResult, ApproachResult
│   ├── mission.py       # Mission, MissionConfig, MissionResult
│   └── experiment.py    # Experiment, ExperimentResult
├── datasets/
│   ├── california_housing.py
│   ├── covertype.py
│   └── synthetic.py
├── models/
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   └── random_forest_model.py
├── tasks/
│   ├── path_dependent_shap.py
│   └── background_shap.py
└── report/
    └── html_generator.py   # Plotly-based self-contained HTML report
benchmarks/
└── example_experiment.py
cache/                      # gitignored — dataset & model cache
results/                    # gitignored — JSON results & HTML reports
```

---

## Adding your own approach

Subclass `Approach`, set `method`, implement `run()`, and optionally `complexity_formula()`:

```python
from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.params import TreeParameters
import time

class MyFastApproach(Approach):
    name    = "my_fast_approach"
    method  = "woodelf"          # or "shap" — used by the scorer
    description = "My optimised implementation."

    def run(self, trained_model, X_explain, X_background):
        t0 = time.perf_counter()
        # ... your implementation ...
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def complexity_formula(self, params: TreeParameters):
        return params.n * params.T * params.D   # unscaled theoretical cost
```

Then pass it to a task factory:

```python
BackgroundSHAPTask(extra_approaches=[MyFastApproach()])
```

---

## Caching

| Cache level | Location | Invalidated by |
|---|---|---|
| Dataset | `cache/datasets/` | `delete_dataset_cache=True` |
| Trained model | `cache/models/` | `delete_model_cache=True` |
| Experiment results | `results/{name}.json` | `delete_results=True` or `force_rerun=True` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `scikit-learn` | Random Forest, synthetic datasets |
| `xgboost`, `lightgbm` | Tree ensemble models |
| `shap>=0.46` | Reference SHAP implementation |
| `woodelf_explainer` | Woodelf SHAP implementation |
| `plotly` | Interactive HTML report |
| `joblib` | Model serialisation |
