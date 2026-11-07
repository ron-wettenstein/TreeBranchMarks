"""
Colab helpers for benchmarks/woodelfhd_m100_experiment.py, used by
colab/woodelfhd_m100_1_woodelf_and_shap.ipynb and colab/woodelfhd_m100_2_pltreeshap.ipynb.

Everything that is expensive to redo is kept on Google Drive (DRIVE_ROOT):

    results/{method}.json      method-cache JSON per method; the experiment rewrites it
                               (atomically) after every finished cell via --result_location
    datasets/{name}/           X.parquet, y.npy, details.json; prepared once, used by both notebooks
    models_main/, models_plt/  LightGBM model caches, one per notebook (the PLTreeSHAP notebook
                               runs Python 3.8 with older libraries, so models are not shared)
    woodelfhd_m100_experiment.html   the final report

Commands
    restore        Drive -> local: datasets, method results; links cache/models to Drive
    save-datasets  download/prepare the datasets locally and copy them to Drive
    progress       finished vs expected cells per method
    report         build the HTML report from the Drive results (runs nothing)

restore and the fallback of progress only need the standard library, so they also work
in the PLTreeSHAP notebook, whose kernel does not have treebranchmarks installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

EXPERIMENT_NAME = "woodelfhd_m100_experiment"
DATASET_FILES = ("X.parquet", "y.npy", "details.json")
DEFAULT_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _has_dataset(d: Path) -> bool:
    return all((d / f).exists() for f in DATASET_FILES)


def _experiment(repo: Path):
    """Build the experiment (requires treebranchmarks and its dependencies)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from benchmarks.woodelfhd_m100_experiment import build_experiment
    return build_experiment()


def _expected_cells(exp) -> dict:
    """method name -> {(mission, D)} for every cell that actually runs (has a model)."""
    from treebranchmarks.core.mission import ModelSpec
    cells = defaultdict(set)
    for ms in exp.missions:
        for ov in ms.approach_overrides:
            for D, spec in ov.model_by_D.items():
                if isinstance(spec, ModelSpec) and D in ms.D_values:
                    cells[ov.approach.method.name].add((ms.name, D))
    return cells


def _done_cells(results_dir: Path) -> dict:
    """method name -> {(mission, D)} found in the method-cache JSONs."""
    done = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            entries = json.load(fh).values()
        done[f.stem] = {(e["_mission"], e["_params"]["D"]) for e in entries if "_params" in e}
    return done


def _print_missing(missing) -> None:
    by_mission = defaultdict(list)
    for mission, D in sorted(missing):
        by_mission[mission].append(D)
    for mission, Ds in by_mission.items():
        print(f"      missing  {mission:40s} D={','.join(map(str, sorted(Ds)))}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def restore(drive_root: Path, repo: Path, methods: list, models_dir: str) -> None:
    cache = repo / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Datasets: copy to local disk (reading 1 GB through the Drive mount on every load is slow).
    ds_drive = drive_root / "datasets"
    on_drive = sorted(d for d in ds_drive.iterdir() if _has_dataset(d)) if ds_drive.exists() else []
    for d in on_drive:
        local = cache / "datasets" / d.name
        if _has_dataset(local):
            print(f"[datasets] {d.name}: already local")
            continue
        for f in DATASET_FILES:
            _copy_atomic(d / f, local / f)
        print(f"[datasets] {d.name}: restored from Drive")
    if not on_drive:
        print("[datasets] none on Drive yet. They are downloaded during the run and NOT kept on Drive, "
              "unless you run notebook 1's 'Prepare datasets' step first.")

    # 2. Models: the local model cache *is* a Drive folder, so trained models survive a new runtime.
    models_drive = drive_root / models_dir
    models_drive.mkdir(parents=True, exist_ok=True)
    link = cache / "models"
    if not link.is_symlink():
        if link.exists():   # models trained before the link existed: keep them
            shutil.copytree(link, models_drive, dirs_exist_ok=True)
            shutil.rmtree(link)
        link.symlink_to(models_drive, target_is_directory=True)
    print(f"[models] {link} -> {models_drive}")

    # 3. Method results: Drive is written together with the local copy, so it is never older.
    local_results = cache / "method_results" / EXPERIMENT_NAME
    local_results.mkdir(parents=True, exist_ok=True)
    for m in methods:
        src = drive_root / "results" / f"{m}.json"
        if not src.exists():
            print(f"[results] {m}: nothing on Drive yet, starting fresh")
            continue
        with open(src) as fh:
            n = len(json.load(fh))   # fail loudly on a corrupt file instead of silently re-running
        _copy_atomic(src, local_results / f"{m}.json")
        print(f"[results] {m}: restored {n} finished cells")


def save_datasets(drive_root: Path, repo: Path) -> None:
    os.chdir(repo)   # the experiment's cache paths are relative to the repo root
    exp = _experiment(repo)
    datasets = {ms.dataset.name: ms.dataset for ms in exp.missions}
    for name, ds in datasets.items():
        drive_dir = drive_root / "datasets" / name
        local_dir = repo / "cache" / "datasets" / name
        if _has_dataset(drive_dir):
            print(f"[datasets] {name}: already on Drive")
            continue
        if not _has_dataset(local_dir):
            print(f"[datasets] {name}: downloading and preprocessing")
            ds.load()
        if not _has_dataset(local_dir):
            print(f"[datasets] {name}: no cache files were written, skipping")
            continue
        for f in DATASET_FILES:
            _copy_atomic(local_dir / f, drive_dir / f)
        print(f"[datasets] {name}: saved to Drive")


def progress(drive_root: Path, repo: Path, methods: list) -> None:
    done = _done_cells(drive_root / "results")
    try:
        expected = _expected_cells(_experiment(repo))
    except Exception:
        expected = None   # experiment not importable in this kernel (PLTreeSHAP notebook): counts only
    for name in methods or sorted(expected or done):
        d = done.get(name, set())
        if expected is None:
            print(f"{name:26s} {len(d):3d} cells done")
            continue
        e = expected.get(name, set())
        print(f"{name:26s} {len(e & d):3d}/{len(e)} cells done")
        _print_missing(e - d)


def report(drive_root: Path, repo: Path) -> None:
    results = drive_root / "results"
    exp = _experiment(repo)
    expected, done = _expected_cells(exp), _done_cells(results)
    missing = {n: e - done.get(n, set()) for n, e in expected.items() if e - done.get(n, set())}
    if missing:
        for n, cells in missing.items():
            print(f"{n}: {len(cells)} cells not finished")
            _print_missing(cells)
        raise SystemExit("Not every cell is finished; the report would try to run them. "
                         "Finish both notebooks first.")

    # Work in an empty folder that holds only the results and the dataset details.json files.
    # With no models on disk, every cell is served from the method cache by its own stored
    # parameters, so nothing is downloaded, trained or timed.
    work = Path(tempfile.mkdtemp(prefix="m100_report_"))
    for f in results.glob("*.json"):
        _copy_atomic(f, work / "cache" / "method_results" / EXPERIMENT_NAME / f.name)
    for name in {ms.dataset.name for ms in exp.missions}:
        details = drive_root / "datasets" / name / "details.json"
        if not details.exists():
            raise SystemExit(f"{details} is missing; run 'save-datasets' first.")
        _copy_atomic(details, work / "cache" / "datasets" / name / "details.json")

    os.chdir(work)   # the experiment's cache/ and results/ paths are relative
    exp.run()
    html = Path(exp.generate_html()).resolve()
    out = drive_root / f"{exp.name}.html"
    _copy_atomic(html, out)
    print(f"Report saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["restore", "save-datasets", "progress", "report"])
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--methods", nargs="*", default=[])
    p.add_argument("--models-dir", default="models_main")
    a = p.parse_args()

    if a.command == "restore":
        restore(a.drive_root, a.repo, a.methods, a.models_dir)
    elif a.command == "save-datasets":
        save_datasets(a.drive_root, a.repo)
    elif a.command == "progress":
        progress(a.drive_root, a.repo, a.methods)
    else:
        report(a.drive_root, a.repo)


if __name__ == "__main__":
    main()
