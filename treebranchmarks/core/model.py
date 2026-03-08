"""
Model management: training, caching, and tree parameter extraction.

Design
------
ModelConfig is a plain value object that fully describes a training run
(ensemble type + hyperparameters + random seed).  Its stable hash is used
as the cache key, so the same config always hits the same cached artifact.

TrainedModel holds the raw sklearn/XGBoost/LightGBM model together with
the TreeParameters extracted from it (T, D, L, F — n and m are left at 0
until task run time).

ModelWrapper is an ABC with one key abstract method per ensemble type:
    - train()                  — fit and return a TrainedModel

The base class provides a default _extract_tree_params() via
woodelf.parse_models.load_decision_tree_ensemble_model, which handles all
supported model types (XGBoost, LightGBM, Random Forest, HistGradientBoosting).

load_or_train() is the public entry point.  It checks the cache first and
only trains if there is a miss.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from woodelf.parse_models import load_decision_tree_ensemble_model

from treebranchmarks.core.params import EnsembleType, TreeParameters, partial_tree_params


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Fully describes a single training configuration.

    Frozen so it can be hashed and used as a cache key.
    Use the `hyperparams` dict for library-specific kwargs
    (e.g. n_estimators, max_depth, learning_rate).
    """

    ensemble_type: EnsembleType
    hyperparams: dict = field(default_factory=dict)
    random_state: int = 42

    def as_dict(self) -> dict:
        return {
            "ensemble_type": self.ensemble_type.value,
            "hyperparams": self.hyperparams,
            "random_state": self.random_state,
        }

    def cache_key(self) -> str:
        """Stable MD5 hash of this config."""
        serialized = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def __str__(self) -> str:
        return f"ModelConfig({self.ensemble_type.value}, {self.hyperparams})"

    # dataclass with a mutable default (dict) requires a custom __hash__
    def __hash__(self) -> int:
        return hash(self.cache_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.cache_key() == other.cache_key()


# ---------------------------------------------------------------------------
# TrainedModel
# ---------------------------------------------------------------------------

@dataclass
class TrainedModel:
    """
    The result of a training run.

    params.n and params.m are 0 here — they are filled in at task run time
    via params.with_run_params(n, m).
    """

    raw_model: Any                  # the actual XGBoost/LightGBM/sklearn model
    config: ModelConfig
    params: TreeParameters          # T, D, L, F set; n=0, m=0
    train_time_s: float
    dataset_name: str


# ---------------------------------------------------------------------------
# ModelWrapper ABC
# ---------------------------------------------------------------------------

class ModelWrapper(ABC):
    """
    Base class for ensemble wrappers.

    Parameters
    ----------
    use_cache : bool
        If True (default), trained models are saved to / loaded from disk.
        Set False for small/fast models to skip the cache overhead.

    Caching design
    --------------
    The cache for a trained model is a directory:
        cache/models/{dataset_name}/{config_hash}/
            meta.json      — ModelConfig + TreeParameters + timing (JSON, human-readable)
            model.*        — the raw model artifact in the library's native format

    The base class owns meta.json serialization/deserialization.
    Subclasses own the raw model artifact via two abstract methods:
        _save_model_artifact(model_dir, raw_model)
        _load_model_artifact(model_dir) -> Any

    Subclasses must implement:
        train()               — fit and return a TrainedModel
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelConfig,
        dataset_name: str,
    ) -> TrainedModel:
        """Fit the model and return a TrainedModel with params populated."""

    def _extract_tree_params(
        self,
        raw_model: Any,
        X: pd.DataFrame,
        config: ModelConfig,
    ) -> TreeParameters:
        """
        Derive T, D, L, F from the fitted model via woodelf's universal parser.

        n and m are left at 0 (populated at task run time).
        """
        ensemble = load_decision_tree_ensemble_model(raw_model, list(X.columns))
        T = len(ensemble.trees)
        D = ensemble.max_depth
        L = float(np.mean([len(tree.get_all_leaves()) for tree in ensemble.trees]))
        F = X.shape[1]
        return partial_tree_params(T=T, D=D, L=L, F=F, ensemble_type=config.ensemble_type)

    @abstractmethod
    def _save_model_artifact(self, model_dir: Path, raw_model: Any) -> None:
        """
        Persist the raw model to model_dir using the library's native format.

        Examples:
          XGBoost      → raw_model.save_model(model_dir / "model.json")
          LightGBM     → raw_model.booster_.save_model(model_dir / "model.txt")
          RandomForest → joblib.dump(raw_model, model_dir / "model.joblib")
        """

    @abstractmethod
    def _load_model_artifact(self, model_dir: Path) -> Any:
        """
        Load and return the raw model from model_dir.

        Must be the inverse of _save_model_artifact().
        """

    # ------------------------------------------------------------------
    # Public entry point — use this instead of train() directly
    # ------------------------------------------------------------------

    def load_or_train(
        self,
        dataset_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelConfig,
        cache_root: Path = Path("cache"),
    ) -> TrainedModel:
        """
        Return a cached TrainedModel if one exists for (dataset, config);
        otherwise train, cache, and return it.
        """
        if self.use_cache:
            model_dir = self._model_cache_dir(cache_root, dataset_name, config)
            if (model_dir / "meta.json").exists():
                print(f"[model:{config.ensemble_type.value}] Cache hit — loading from {model_dir.name}/")
                return self._load_cached_model(model_dir)

        print(f"[model:{config.ensemble_type.value}] Training.")
        trained = self.train(X, y, config, dataset_name)
        print(
            f"[model:{config.ensemble_type.value}] Trained in {trained.train_time_s:.2f}s — "
            f"T={trained.params.T}, D={trained.params.D}, L={trained.params.L:.1f}, F={trained.params.F}"
        )

        if self.use_cache:
            self._save_model(model_dir, trained)

        return trained

    # ------------------------------------------------------------------
    # Cache internals — base class manages meta.json, subclass manages artifact
    # ------------------------------------------------------------------

    def _model_cache_dir(
        self,
        cache_root: Path,
        dataset_name: str,
        config: ModelConfig,
    ) -> Path:
        model_dir = cache_root / "models" / dataset_name / config.cache_key()
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _save_model(self, model_dir: Path, trained: TrainedModel) -> None:
        # 1. Save the raw model in its native format.
        self._save_model_artifact(model_dir, trained.raw_model)

        # 2. Save everything else as a human-readable JSON sidecar.
        meta = {
            "config": trained.config.as_dict(),
            "params": trained.params.as_dict(),
            "train_time_s": trained.train_time_s,
            "dataset_name": trained.dataset_name,
        }
        with open(model_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _load_cached_model(self, model_dir: Path) -> TrainedModel:
        # 1. Load the raw model from its native format.
        raw_model = self._load_model_artifact(model_dir)

        # 2. Reconstruct TrainedModel from the JSON sidecar.
        with open(model_dir / "meta.json") as f:
            meta = json.load(f)

        c = meta["config"]
        config = ModelConfig(
            ensemble_type=EnsembleType(c["ensemble_type"]),
            hyperparams=c["hyperparams"],
            random_state=c["random_state"],
        )
        p = meta["params"]
        params = TreeParameters(
            n=p["n"], m=p["m"], F=p["F"],
            T=p["T"], D=p["D"], L=p["L"],
            ensemble_type=EnsembleType(p["ensemble_type"]),
        )
        return TrainedModel(
            raw_model=raw_model,
            config=config,
            params=params,
            train_time_s=meta["train_time_s"],
            dataset_name=meta["dataset_name"],
        )
