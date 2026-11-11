"""
Colab helpers for benchmarks/woodelfhd_m100_gpu_experiment.py, used by
colab/woodelfhd_m100_3_woodelfhd_gpu.ipynb.

The same commands as woodelfhd_m100_tools.py, pointed at the GPU experiment, plus check-models.
The GPU notebook shares notebook 1's Drive folder:

    datasets/        read (prepared by notebook 1)
    models_main/     read: the GPU cells load the models notebook 1 trained for WoodelfHD
    results/         woodelf_hd_gpu.json is written here; the CPU results next to it are only read
    woodelfhd_m100_gpu_experiment.html   the report, CPU and GPU rows side by side

Commands
    restore        Drive -> local: datasets, results; links cache/models to Drive
    check-models   stop unless every GPU cell's model is already in the Drive model cache
    progress       finished vs expected cells per method
    report         build the HTML report from the Drive results (runs nothing)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import woodelfhd_m100_tools as base  # noqa: E402

GPU_METHOD = "woodelf_hd_gpu"
base.EXPERIMENT_NAME = "woodelfhd_m100_gpu_experiment"


def _experiment(repo: Path):
    """Build the GPU experiment (requires treebranchmarks and its dependencies)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from benchmarks.woodelfhd_m100_gpu_experiment import build_experiment
    return build_experiment()


base._experiment = _experiment   # restore / progress / report look it up at call time


def check_models(repo: Path) -> None:
    """Every GPU cell must load a model that notebook 1 already trained (same file as the CPU cell)."""
    from treebranchmarks.core.mission import ModelSpec

    os.chdir(repo)   # the experiment's cache paths are relative to the repo root
    exp = _experiment(repo)
    found, missing = 0, []
    for ms in exp.missions:
        for ov in ms.approach_overrides:
            if ov.approach.method.name != GPU_METHOD:
                continue
            for D, spec in sorted(ov.model_by_D.items()):
                if D not in ms.D_values or not isinstance(spec, ModelSpec):
                    continue
                if spec.wrapper.load_params_only(ms.cache_root, ms.dataset.name, spec.config) is None:
                    missing.append((ms.name, D, spec.config.hyperparams.get("n_estimators")))
                else:
                    found += 1
    if missing:
        for name, D, T in missing:
            print(f"      missing model  {name:40s} D={D} T={T}")
        raise SystemExit(
            f"{len(missing)} models are not in the Drive model cache. Run WoodelfHD in notebook 1 "
            "first: the GPU cells must use the same models as the CPU cells.")
    print(f"[models] all {found} models of the GPU cells are in the Drive model cache")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check-models":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("command")
        p.add_argument("--repo", type=Path, default=base.DEFAULT_REPO)
        check_models(p.parse_args().repo)
    else:
        base.main()
