# treebranchmarks

A benchmarking framework for comparing SHAP algorithms on decision tree ensembles.

Measures and visualises how runtime scales across the six complexity parameters:

| Parameter | Meaning |
|---|---|
| `n` | rows being explained |
| `m` | rows in the background dataset |
| `T` | number of trees |
| `D` | tree depth |
| `L` | average leaves per tree |
| `F` | number of features |

Results are summarised in a self-contained interactive HTML report with log-scale charts, a full results table, and a head-to-head scoreboard.

---

## Installation

```bash
pip install -e .
pip install -e ".[dev]"   # include pytest
```

Python 3.10+ required.

---

## Quick start

```bash
python -m benchmarks.example_experiment
```

Or build your own:

```python
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.datasets import CaliforniaHousingDataset
from treebranchmarks.models import LightGBMWrapper
from treebranchmarks.methods import SHAPApproach, WoodelfApproach

experiment = Experiment(
    name="my_benchmark",
    missions=[
        Mission(MissionConfig(
            dataset=CaliforniaHousingDataset(),
            model_wrappers={
                ModelConfig(EnsembleType.LIGHTGBM, {"n_estimators": 100, "max_depth": 6}):
                LightGBMWrapper()
            },
            tasks=[Task(TaskType.PATH_DEPENDENT_SHAP, [SHAPApproach(), WoodelfApproach()])],
            n_values=[10, 100, 1_000, 10_000],
            m_values=[0],
        )),
    ],
)
experiment.run()
experiment.generate_html()
```

---

## Mission types

### `Mission` — standard parameter sweep

Sweeps `n_values`, `m_values`, or a set of model configs (varying `D`). Iterates smallest-to-largest; if the linearly-extrapolated runtime exceeds `timeout_s` (default 600 s) the run is skipped and flagged `is_estimated=True`.

### `ControlledMission` — per-approach, per-D model

For D-sweep experiments where different approaches need different models at each depth (e.g. one approach uses 1 tree at D=30, another crashes):

```python
from treebranchmarks.core.mission import (
    ControlledMission, ApproachDOverride, ModelSpec, MEMORY_CRASH, PrerecordedTime
)

mission = ControlledMission(
    name="D sweep",
    dataset=FraudDetectionDataset(),
    D_values=[6, 9, 12, 20],
    approach_overrides=[
        ApproachDOverride(
            approach=WoodelfApproach(),
            full_T=100,
            model_by_D={
                6:  ModelSpec(lgbm_d6_100t, LightGBMWrapper()),
                9:  ModelSpec(lgbm_d9_10t,  LightGBMWrapper()),  # scaled ×10
                12: ModelSpec(lgbm_d12_1t,  LightGBMWrapper()),  # scaled ×100
                20: MEMORY_CRASH,
            },
        ),
        ApproachDOverride(
            approach=SHAPApproach(),
            full_T=100,
            model_by_D={
                6:  ModelSpec(lgbm_d6_100t, LightGBMWrapper()),
                20: PrerecordedTime(elapsed_s=3600.0, estimation_description="pre-run"),
            },
        ),
    ],
    task_types=[TaskType.PATH_DEPENDENT_SHAP],
    n=1000, m=0,
)
```

- `MEMORY_CRASH` → records a memory-crash result for that D.
- `PrerecordedTime` → injects a known elapsed time without re-running.
- When `actual_T < full_T`, elapsed time is scaled by `full_T / actual_T` and `is_estimated=True`.

---

## CLI runner

Every experiment supports CLI flags via `run_experiment_cli`:

```bash
# Run only Woodelf approaches
python -m benchmarks.fraud_depth_experiment --method woodelf

# Run multiple methods, save results to an extra path
python -m benchmarks.fraud_depth_experiment --method woodelf --method shap \
    --result_location /tmp/partial_results.json
```

- `--method` filters by `approach.method.name` (case-insensitive, repeatable).
- `--result_location` dual-writes the results JSON after every completed mission, so partial results are preserved if the run is interrupted.

Results are also written incrementally to `results/{name}.json` after each mission regardless of these flags.

---

## Datasets

| Class | Source |
|---|---|
| `CaliforniaHousingDataset` | sklearn built-in |
| `SyntheticDataset` | sklearn `make_classification` |
| `CovertypeDataset` | sklearn built-in |
| `BreastCancerDataset` | sklearn built-in |
| `FraudDetectionDataset` | Kaggle IEEE-CIS |
| `IntrustionDetectionDataset` | KDD-99 |
| `HIGGSDataset` | UCI (downloaded automatically) |

---

## Caching

| Cache level | Location | Invalidated by |
|---|---|---|
| Dataset | `cache/datasets/` | `delete_dataset_cache=True` |
| Trained model | `cache/models/` | `delete_model_cache=True` |
| Per-method results | `cache/method_results/` | `force_rerun_methods=[...]` |
| Experiment results | `results/{name}.json` | `force_rerun=True` or `delete_results=True` |

Per-method caching means re-running a single approach (e.g. after adding a new one) does not re-time the others.

---

## Adding a new approach

Subclass `Approach`, set `method`, implement the task methods you support:

```python
from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.method import Method
import time

MY_METHOD = Method(name="my_method", label="My Method", description="...")

class MyApproach(Approach):
    name        = "My Approach"
    method      = MY_METHOD
    description = "My optimised implementation."

    def path_dependent_shap(self, trained_model, X_explain, X_background):
        t0 = time.perf_counter()
        # ... your implementation ...
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)
```

---

## Dependencies

`numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `shap>=0.46`, `woodelf_explainer`, `plotly`, `joblib`
