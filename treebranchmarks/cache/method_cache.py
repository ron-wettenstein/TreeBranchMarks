"""
MethodResultCache: persists ApproachResult data per method.

Layout on disk
--------------
cache/method_results/{experiment_name}/{method_name}.json

Each file is a JSON object mapping a stable group key to an ApproachResult dict.
The group key encodes (mission, task, n, m, D, T, L, ensemble, approach_name) so
results are invalidated automatically when model hyperparams or data sizes change.

Usage
-----
cache = MethodResultCache(experiment_name="my_exp", cache_root=Path("cache"))

# Before running an approach:
cached = cache.get(approach, mission_name, task_name, params)
if cached is not None:
    return cached          # skip the actual benchmark run

# After running:
cache.put(approach, mission_name, task_name, params, approach_result)

# Clear one method (call before re-running that method):
cache.clear_method(method_name)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from treebranchmarks.core.approach import Approach
    from treebranchmarks.core.params import TreeParameters
    from treebranchmarks.core.task import ApproachResult


def _group_key(
    approach_name: str,
    mission_name: str,
    task_name: str,
    params: "TreeParameters",
) -> str:
    """Stable MD5 key for one (approach, mission, task, params) combination."""
    payload = json.dumps({
        "approach": approach_name,
        "mission": mission_name,
        "task": task_name,
        "n": params.n,
        "m": params.m,
        "D": params.D,
        "T": params.T,
        "L": round(params.L, 4),
        "F": params.F,
        "ensemble": params.ensemble_type.value,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


class MethodResultCache:
    """
    Per-method JSON cache for ApproachResult objects.

    Parameters
    ----------
    experiment_name : str
        Used to namespace the cache directory.
    cache_root : Path
        Root cache directory (default ``cache/``).
    """

    def __init__(
        self,
        experiment_name: str,
        cache_root: Path = Path("cache"),
        extra_paths: Optional[list[Path]] = None,
    ) -> None:
        self._dir = cache_root / "method_results" / experiment_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._extra_paths: list[Path] = list(extra_paths or [])
        # In-memory store:  method_name → { key: ApproachResult-dict }
        self._data: dict[str, dict[str, dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        approach: "Approach",
        mission_name: str,
        task_name: str,
        params: "TreeParameters",
    ) -> Optional["ApproachResult"]:
        """Return a cached ApproachResult, or None if not cached."""
        method_name = self._method_name(approach)
        if method_name is None:
            return None
        store = self._load(method_name)
        key = _group_key(approach.name, mission_name, task_name, params)
        raw = store.get(key)
        if raw is None:
            return None
        return self._deserialize(raw)

    def put(
        self,
        approach: "Approach",
        mission_name: str,
        task_name: str,
        params: "TreeParameters",
        result: "ApproachResult",
    ) -> None:
        """Store an ApproachResult in the cache and flush to disk."""
        method_name = self._method_name(approach)
        if method_name is None:
            return
        store = self._load(method_name)
        key = _group_key(approach.name, mission_name, task_name, params)
        entry = result.as_dict()
        entry["_label"] = f"{task_name} n={params.n} m={params.m} D={params.D} T={params.T}"
        entry["_mission"] = mission_name
        entry["_task"] = task_name
        entry["_params"] = {
            "n": params.n, "m": params.m, "D": params.D, "T": params.T,
            "L": round(params.L, 4), "F": params.F,
            "ensemble": params.ensemble_type.value,
        }
        store[key] = entry
        self._flush(method_name, store)

    def all_approaches_cached(
        self,
        approaches: list["Approach"],
        mission_name: str,
        task_name: str,
        params: "TreeParameters",
    ) -> bool:
        """Return True if every approach has a cached result for the given params."""
        return all(
            self.get(approach, mission_name, task_name, params) is not None
            for approach in approaches
        )

    def recover_params(
        self,
        approach: "Approach",
        mission_name: str,
        task_name: str,
        D: int,
        T: int,
        n: int,
        m: int,
        ensemble: str,
    ) -> "Optional[TreeParameters]":
        """
        Find TreeParameters for a cached entry by identity fields (D, T, n, m, ensemble),
        without needing the model files.  Returns None if no matching entry is found or
        if the entry was written before the _params field was added.

        Used by ControlledMission when load_params_only() returns None (model not yet
        cached locally, e.g. after a Colab session restart).
        """
        from treebranchmarks.core.params import EnsembleType, TreeParameters

        method_name = self._method_name(approach)
        if method_name is None:
            return None
        store = self._load(method_name)
        for raw in store.values():
            p = raw.get("_params")
            if (
                p is None
                or raw.get("_mission") != mission_name
                or raw.get("_task") != task_name
                or p["D"] != D or p["T"] != T
                or p["n"] != n or p["m"] != m
                or p["ensemble"] != ensemble
            ):
                continue
            return TreeParameters(
                n=p["n"], m=p["m"], F=p["F"],
                T=p["T"], D=p["D"], L=p["L"],
                ensemble_type=EnsembleType(p["ensemble"]),
            )
        return None

    def clear_method(self, method_name: str) -> None:
        """Delete the cache file for one method and evict from memory."""
        path = self._path(method_name)
        if path.exists():
            path.unlink()
            print(f"[method-cache] Cleared cache for method '{method_name}'")
        self._data.pop(method_name, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _method_name(approach: "Approach") -> Optional[str]:
        m = getattr(approach, "method", None)
        if m is None:
            return None
        # Support both Method objects and plain strings (fallback)
        return getattr(m, "name", m) or None

    def _path(self, method_name: str) -> Path:
        return self._dir / f"{method_name}.json"

    def _load(self, method_name: str) -> dict:
        if method_name not in self._data:
            path = self._path(method_name)
            if path.exists():
                with open(path) as f:
                    self._data[method_name] = json.load(f)
            else:
                self._data[method_name] = {}
        return self._data[method_name]

    def _flush(self, method_name: str, store: dict) -> None:
        data = json.dumps(store, indent=2)
        with open(self._path(method_name), "w") as f:
            f.write(data)
        for extra in self._extra_paths:
            extra.parent.mkdir(parents=True, exist_ok=True)
            with open(extra, "w") as f:
                f.write(data)

    @staticmethod
    def _deserialize(raw: dict) -> "ApproachResult":
        from treebranchmarks.core.task import ApproachResult
        return ApproachResult(
            approach_name=raw["approach_name"],
            running_time=raw["running_time"],
            std_time_s=raw["std_time_s"],
            is_estimated=raw["is_estimated"],
            error=raw.get("error"),
            method=raw.get("method", ""),
            not_supported=raw.get("not_supported", False),
            memory_crash=raw.get("memory_crash", False),
            runtime_error=raw.get("runtime_error", False),
            estimation_description=raw.get("estimation_description", ""),
        )
