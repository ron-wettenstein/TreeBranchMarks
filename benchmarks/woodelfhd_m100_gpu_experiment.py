"""
WoodelfHD Depth Sweep, m=100 background, plus WoodelfHD on GPU.

This experiment is ``benchmarks/woodelfhd_m100_experiment.py`` with one more approach,
WoodelfHD GPU. It is built from that experiment's own missions, so everything that decides
the numbers is shared with the CPU rows:

- Datasets, depths and LightGBM configs (and therefore the model cache keys). The Colab notebook
  links cache/models to the Drive folder that notebook 1 trained into, so the GPU cells load the
  very same model files as the CPU cells.
- Explained and background rows. Each mission draws them with ``default_rng(42)``, once per
  depth and in the same order, whichever approaches run.
- Trees per cell. The GPU approach gets WoodelfHD's (CPU) ``model_by_D``, so every GPU cell runs
  on the same number of trees as the CPU cell, and gets the same estimation mark.

Only the GPU approach is meant to run here (``--method woodelf_hd_gpu``). The other approaches'
results come from the m=100 method-cache files on Drive, so the report built by
``colab/woodelfhd_m100_gpu_tools.py report`` shows the CPU and GPU rows side by side.

GPU timing
----------
``SyncedWoodelfHDGPUApproach`` calls ``woodelf_for_high_depth(..., GPU=True)`` exactly like
``WoodelfHDGPUApproach``, with two additions:

1. Warmup. Before the first timed call of each task in the process, the same call is run once,
   untimed, on a small slice of the inputs. This pays the one-time costs of the process (CUDA
   context, library handles, CuPy kernel compilation), which would otherwise land on the first
   timed cell. Every cell is timed only once, so there is no other place they could average out.
2. Synchronization. ``woodelf_for_high_depth(GPU=True)`` returns CuPy arrays without waiting for
   the GPU, so the clock is stopped only after ``cupy.cuda.Device().synchronize()``.

The CuPy memory pools are emptied after each call, outside the timed region.

Run from project root (on a machine with an NVIDIA GPU and CuPy):
    python -m benchmarks.woodelfhd_m100_gpu_experiment --method woodelf_hd_gpu
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd
from woodelf.core.cube_metric import ShapleyInteractionValues, ShapleyValues
from woodelf.high_depth_woodelf import woodelf_for_high_depth

from treebranchmarks import Experiment
from treebranchmarks.core.approach import ApproachOutput
from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks.core.mission import ControlledMission
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import WOODELF_HD
from treebranchmarks.methods.woodelf_original_and_hd_method import WoodelfHDGPUApproach

from benchmarks import woodelfhd_m100_experiment as m100

EXPERIMENT_NAME = "woodelfhd_m100_gpu_experiment"
CACHE_ROOT = m100.CACHE_ROOT
RESULTS_DIR = m100.RESULTS_DIR

_WARMUP_EXPLAIN_ROWS = 256
_WARMUP_BACKGROUND_ROWS = 16


class SyncedWoodelfHDGPUApproach(WoodelfHDGPUApproach):
    """WoodelfHD GPU with a one-time warmup per task and a synchronized clock (see module docstring)."""

    description = (
        "woodelf_for_high_depth accelerated on GPU (CuPy). One untimed warmup call per task per "
        "process; the clock stops after the GPU has synchronized."
    )

    # (metric, has background) pairs already warmed up in this process.
    _warmed_up: set = set()

    @staticmethod
    def _cupy():
        import cupy  # imported lazily so the experiment can be built on a machine without a GPU
        return cupy

    def _timed(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        metric_cls,
    ) -> ApproachOutput:
        cp = self._cupy()
        model = trained_model.raw_model

        key = (metric_cls.__name__, X_background is not None)
        if key not in self._warmed_up:
            warm_bg = None if X_background is None else X_background.iloc[:_WARMUP_BACKGROUND_ROWS]
            woodelf_for_high_depth(model, X_explain.iloc[:_WARMUP_EXPLAIN_ROWS], warm_bg, metric_cls(), GPU=True)
            cp.cuda.Device().synchronize()
            type(self)._warmed_up.add(key)
            self._free_pools(cp)

        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        woodelf_for_high_depth(model, X_explain, X_background, metric_cls(), GPU=True)
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - t0

        self._free_pools(cp)
        return ApproachOutput(elapsed_s=elapsed)

    @staticmethod
    def _free_pools(cp) -> None:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def path_dependent_shap(self, trained_model, X_explain, X_background):
        return self._timed(trained_model, X_explain, None, ShapleyValues)

    def path_dependent_interactions(self, trained_model, X_explain, X_background):
        return self._timed(trained_model, X_explain, None, ShapleyInteractionValues)

    def background_shap(self, trained_model, X_explain, X_background):
        return self._timed(trained_model, X_explain, X_background, ShapleyValues)

    def background_shap_interactions(self, trained_model, X_explain, X_background):
        return self._timed(trained_model, X_explain, X_background, ShapleyInteractionValues)


GPU_APPROACH = SyncedWoodelfHDGPUApproach()


def build_missions(cache_root: Path = CACHE_ROOT) -> list[ControlledMission]:
    """The m=100 missions, each with a WoodelfHD GPU override that mirrors the WoodelfHD (CPU) one."""
    missions = m100.build_missions(cache_root)
    for ms in missions:
        cpu = [ov for ov in ms.approach_overrides if ov.approach.method.name == WOODELF_HD.name]
        if len(cpu) != 1:
            raise RuntimeError(f"{ms.name}: expected exactly one WoodelfHD override, found {len(cpu)}")
        ms.approach_overrides.append(
            replace(cpu[0], approach=GPU_APPROACH, model_by_D=dict(cpu[0].model_by_D))
        )
    return missions


_SUMMARY_HTML = (
    Path(__file__).parent.parent / "treebranchmarks" / "report" / "benchmark_summaries"
    / "woodelfhd_m100_gpu_summary.html"
)


def build_experiment() -> Experiment:
    return Experiment(
        name=EXPERIMENT_NAME,
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
        summary_html_path=_SUMMARY_HTML,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
