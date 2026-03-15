"""
Experiment: the top-level orchestrator.

An Experiment holds a list of Missions and is responsible for:
  - Running them (or loading cached results)
  - Persisting results to JSON under results/{name}.json
  - Delegating HTML generation to HtmlGenerator

Results JSON schema
-------------------
{
  "experiment_name": "...",
  "missions": [
    {
      "dataset": "...",
      "task_results": [
        {
          "task_name": "...",
          "params": { n, m, F, T, D, L, ensemble_type },
          "approach_results": {
            "approach_name": {
              "measured_times_s": [...],
              "mean_time_s": ...,
              "std_time_s": ...,
              "estimated_time_s": ...,
              "error": null
            }
          }
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import shutil
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from treebranchmarks.cache.method_cache import MethodResultCache
from treebranchmarks.core.mission import ControlledMission, Mission, MissionResult
from treebranchmarks.core.params import EnsembleType, TreeParameters
from treebranchmarks.core.task import ApproachResult, TaskResult


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    experiment_name: str
    mission_results: list[MissionResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "missions": [m.as_dict() for m in self.mission_results],
        }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """
    Top-level orchestrator.

    Parameters
    ----------
    name : str
        Unique name for this experiment.  Used as the stem of the results
        file and the HTML report.
    missions : list[Mission]
        The parameter sweeps to execute.
    results_dir : Path
        Where to write results JSON and HTML.  Created if it doesn't exist.
    force_rerun : bool
        If True, ignore the top-level results JSON and re-run all methods.
        Approach-level method caches are still used unless force_rerun_methods
        is also set.
    force_rerun_methods : list | None
        List of Method objects (or method name strings) whose cached approach
        results should be discarded and re-measured.  All other methods reuse
        their cached times.  Use ``force_rerun=True`` to bypass all caches.
    delete_dataset_cache : bool
        Delete the dataset cache before running.
    delete_model_cache : bool
        Delete the model cache before running.
    delete_results : bool
        Delete the top-level results JSON and HTML before running.
    """

    def __init__(
        self,
        name: str,
        missions: list[Mission],
        results_dir: Path = Path("results"),
        force_rerun: bool = False,
        force_rerun_methods: Optional[list] = None,
        delete_dataset_cache: bool = False,
        delete_model_cache: bool = False,
        delete_results: bool = False,
        method_filter: Optional[list[str]] = None,
        extra_result_paths: Optional[list[Path]] = None,
    ) -> None:
        self.name = name
        self.missions = missions
        self.results_dir = results_dir
        self.force_rerun = force_rerun
        self.force_rerun_methods: list[str] = [
            (getattr(m, "name", m) or m) for m in (force_rerun_methods or [])
        ]
        self.delete_dataset_cache = delete_dataset_cache
        self.delete_model_cache = delete_model_cache
        self.delete_results = delete_results
        self.method_filter: list[str] = [m.lower() for m in (method_filter or [])]
        self.extra_result_paths: list[Path] = list(extra_result_paths or [])

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """
        Run all missions and persist results to JSON.

        Uses a per-method approach cache (``cache/method_results/``) so that
        individual methods can be re-run without re-timing the others.
        Set ``force_rerun_methods`` on the Experiment to clear specific method
        caches before running.
        """
        self._maybe_clear_caches()

        # Build the method cache (shared across all missions).
        cache_root = (
            self.missions[0].cache_root
            if self.missions
            else Path("cache")
        )
        method_cache = MethodResultCache(
            experiment_name=self.name,
            cache_root=cache_root,
        )
        for method_name in self.force_rerun_methods:
            method_cache.clear_method(method_name)

        results_path = self._results_path()

        if not self.force_rerun and not self.force_rerun_methods and results_path.exists():
            print(f"[experiment:{self.name}] Loading cached results from {results_path}")
            return self.load_results()

        result = ExperimentResult(experiment_name=self.name)
        for mission in self.missions:
            filtered = self._filter_mission(mission)
            mission_result = filtered.run(method_cache=method_cache)
            result.mission_results.append(mission_result)
            self._persist(result)

        print(f"\n[experiment:{self.name}] Results saved to {results_path}")
        return result

    def load_results(self) -> ExperimentResult:
        """Load previously persisted results from the JSON file."""
        results_path = self._results_path()
        if not results_path.exists():
            raise FileNotFoundError(
                f"No results file found at {results_path}. Run the experiment first."
            )
        with open(results_path) as f:
            raw = json.load(f)
        return self._deserialize(raw)

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def generate_html(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate an interactive HTML report from the persisted results.

        If output_path is not provided, writes to results/{name}.html.
        Returns the path of the generated file.
        """
        from treebranchmarks.report.html_generator import HtmlGenerator

        if output_path is None:
            output_path = self.results_dir / f"{self.name}.html"

        result = self.load_results()
        generator = HtmlGenerator()
        generator.generate(result, output_path)
        print(f"[experiment:{self.name}] HTML report written to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Cache clearing
    # ------------------------------------------------------------------

    def _maybe_clear_caches(self) -> None:
        if self.delete_dataset_cache:
            seen_datasets: set[str] = set()
            for mission in self.missions:
                ds = mission.config.dataset
                if ds.name not in seen_datasets:
                    seen_datasets.add(ds.name)
                    ds.invalidate_cache()

        if self.delete_model_cache:
            seen_dirs: set[Path] = set()
            for mission in self.missions:
                cache_root = mission.config.cache_root
                for model_config in mission.config.model_wrappers:
                    ds_name = mission.config.dataset.name
                    model_dir = cache_root / "models" / ds_name / model_config.cache_key()
                    if model_dir not in seen_dirs and model_dir.exists():
                        seen_dirs.add(model_dir)
                        shutil.rmtree(model_dir)
                        print(f"[experiment:{self.name}] Deleted model cache: {model_dir}")

        if self.delete_results:
            for suffix in (".json", ".html"):
                path = self.results_dir / f"{self.name}{suffix}"
                if path.exists():
                    path.unlink()
                    print(f"[experiment:{self.name}] Deleted: {path}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _results_path(self) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir / f"{self.name}.json"

    def _filter_mission(self, mission):
        """
        Return a (possibly shallow-copied) mission with approaches filtered to
        those whose ``method.name`` is in ``self.method_filter``.

        If ``method_filter`` is empty, the original mission is returned unchanged.
        Works for both ``Mission`` and ``ControlledMission``.
        """
        if not self.method_filter:
            return mission

        def _method_matches(approach) -> bool:
            name = getattr(getattr(approach, "method", None), "name", "")
            return name.lower() in self.method_filter

        if isinstance(mission, ControlledMission):
            filtered_overrides = [
                ao for ao in mission.approach_overrides
                if _method_matches(ao.approach)
            ]
            new_mission = copy(mission)
            new_mission.approach_overrides = filtered_overrides
            return new_mission

        # Standard Mission — filter tasks
        filtered_tasks = []
        for task in mission.config.tasks:
            filtered_approaches = [a for a in task.approaches if _method_matches(a)]
            if filtered_approaches:
                new_task = copy(task)
                new_task.approaches = filtered_approaches
                filtered_tasks.append(new_task)

        new_config = copy(mission.config)
        new_config.tasks = filtered_tasks
        new_mission = copy(mission)
        new_mission.config = new_config
        return new_mission

    def _persist(self, result: ExperimentResult) -> None:
        data = json.dumps(result.as_dict(), indent=2)
        with open(self._results_path(), "w") as f:
            f.write(data)
        for path in self.extra_result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(data)

    def _deserialize(self, raw: dict) -> ExperimentResult:
        """
        Reconstruct an ExperimentResult from the persisted JSON dict.

        We reconstruct only the data needed for reporting (TaskResult /
        ApproachResult / TreeParameters) — we do NOT reload the trained
        models or dataset objects.
        """
        mission_results = []
        for mission_dict in raw.get("missions", []):
            task_results = []
            for tr in mission_dict.get("task_results", []):
                p = tr["params"]
                params = TreeParameters(
                    n=p["n"], m=p["m"], F=p["F"],
                    T=p["T"], D=p["D"], L=p["L"],
                    ensemble_type=EnsembleType(p["ensemble_type"]),
                )
                approach_results = {
                    name: ApproachResult(
                        approach_name=name,
                        running_time=ar["running_time"],
                        std_time_s=ar["std_time_s"],
                        is_estimated=ar["is_estimated"],
                        error=ar.get("error"),
                        method=ar.get("method", ""),
                        not_supported=ar.get("not_supported", False),
                        memory_crash=ar.get("memory_crash", False),
                        runtime_error=ar.get("runtime_error", False),
                        estimation_description=ar.get("estimation_description", ""),
                    )
                    for name, ar in tr["approach_results"].items()
                }
                task_results.append(TaskResult(
                    task_name=tr["task_name"],
                    params=params,
                    approach_results=approach_results,
                ))
            # MissionResult needs a config reference; for deserialized results
            # we only need the data, so we store None and use dataset/mission names.
            mr = MissionResult.__new__(MissionResult)
            mr.config = None  # type: ignore[assignment]
            mr._dataset_name = mission_dict.get("dataset", "unknown")
            mr.mission_name = mission_dict.get("mission_name", "unknown")
            mr.meta = mission_dict.get("meta", {})
            mr.task_results = task_results
            mission_results.append(mr)

        return ExperimentResult(
            experiment_name=raw["experiment_name"],
            mission_results=mission_results,
        )
