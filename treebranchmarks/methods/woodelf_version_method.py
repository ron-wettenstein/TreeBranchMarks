"""
WoodelfVersionApproach — runs a specific git tag/branch of woodelf in a subprocess.

Each approach instance installs the specified ref of woodelf into an isolated
target directory (cache/woodelf_versions/{ref}/), then runs the benchmark task
in a subprocess with that directory prepended to PYTHONPATH.  The host Python
executable is reused so all model libraries (XGBoost, LightGBM, etc.) are
available without reinstalling.

Timing is measured inside the subprocess (around only the shap call), so
subprocess startup and data serialization do not inflate the reported time.

Depth limits are inferred automatically from the git ref:

  Branch (non-tag)     → no limits
  Tag >= 0.4.3         → no limits
  Tag 0.4.1 – 0.4.2   → BG SHAP + BG IV limited; PD unlimited
  Tag 0.2.16 – 0.4.0  → BG SHAP + PD/BG IV limited; PD SHAP unlimited
  Tag 0.2.9 – 0.2.15  → all tasks limited at D=18/20 (PD) or D=15/18 (IV)
  Tag <= 0.2.8         → all tasks crash at D=11

Usage
-----
    from treebranchmarks.methods.woodelf_version_method import WoodelfVersionApproach

    approaches = [
        WoodelfVersionApproach("v0.2.8"),
        WoodelfVersionApproach("v0.4.2"),
        WoodelfVersionApproach("v0.4.3"),
        WoodelfVersionApproach("feature/sparse"),
    ]
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import WOODELF

_WOODELF_REPO = "https://github.com/ron-wettenstein/woodelf.git"
_RUNNER = Path(__file__).parent / "_woodelf_runner.py"

# (tree_limit_depth, memory_crash_depth) — None means no limit
_Limits = tuple[Optional[int], Optional[int]]
_NO_LIMIT: _Limits = (None, None)


# ---------------------------------------------------------------------------
# Depth-limit configuration per task
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DepthLimits:
    pd_shap: _Limits = _NO_LIMIT
    pd_iv:   _Limits = _NO_LIMIT
    bg_shap: _Limits = _NO_LIMIT
    bg_iv:   _Limits = _NO_LIMIT


_TAG_RE = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)$')


def _parse_tag(ref: str) -> Optional[tuple[int, int, int]]:
    """Return (major, minor, patch) if ref looks like a version tag, else None."""
    m = _TAG_RE.match(ref)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _limits_for_ref(ref: str) -> _DepthLimits:
    """
    Return the depth limits that match the historical capabilities of this ref.

    Branches (non-tags) and tags >= 0.4.3 have no limits.
    Older tags progressively add limits as more depth ranges were unsupported.
    """
    version = _parse_tag(ref)

    if version is None or version >= (0, 4, 3):
        return _DepthLimits(bg_iv=(15, 18))

    if version >= (0, 4, 2):
        # BG tasks limited; PD SHAP and PD IV are unlimited in this range
        return _DepthLimits(
            bg_shap=(18, 20),
            bg_iv=(15, 18),
        )

    if version >= (0, 2, 16):
        # PD IV now also limited
        return _DepthLimits(
            pd_iv=(15, 18),
            bg_shap=(18, 20),
            bg_iv=(15, 18),
        )

    if version >= (0, 2, 9):
        # PD SHAP also limited
        return _DepthLimits(
            pd_shap=(18, 20),
            pd_iv=(15, 18),
            bg_shap=(18, 20),
            bg_iv=(15, 18),
        )

    # <= 0.2.8: everything crashes at D=11
    return _DepthLimits(
        pd_shap=(11, 11),
        pd_iv=(11, 11),
        bg_shap=(11, 11),
        bg_iv=(11, 11),
    )


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------

class WoodelfVersionApproach(Approach):
    """
    Runs WoodelfExplainer from a specific git tag or branch in a subprocess.

    Parameters
    ----------
    ref : str
        A git tag or branch name (e.g. "v1.2.0", "feature/sparse").
    repo : str
        Git URL or local path to the woodelf repository.
    cache_root : Path
        Root of the treebranchmarks cache (default "cache").
    gpu : bool
        Passed through to WoodelfExplainer(GPU=...).
    """

    method = WOODELF

    def __init__(
        self,
        ref: str,
        repo: str = _WOODELF_REPO,
        cache_root: Path = Path("cache"),
        gpu: bool = False,
    ) -> None:
        self.ref = ref
        self.repo = repo
        self._cache_root = Path(cache_root)
        self.GPU = gpu
        self.name = f"Woodelf@{ref}"
        self.description = f"Woodelf from git ref '{ref}'."
        self._limits = _limits_for_ref(ref)
        self._install_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def _ensure_install(self) -> Path:
        """Return the --target directory for this ref, installing if needed."""
        if self._install_dir is not None:
            return self._install_dir

        ref_slug = self.ref.replace("/", "_").replace("\\", "_")
        install_dir = self._cache_root / "woodelf_versions" / ref_slug
        marker = install_dir / ".installed"

        if not marker.exists():
            install_dir.mkdir(parents=True, exist_ok=True)
            print(f"[woodelf_version] Installing woodelf@{self.ref} into {install_dir} ...")
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "--target", str(install_dir),
                    "--no-deps",
                    f"git+{self.repo}@{self.ref}",
                ],
                check=True,
            )
            marker.write_text(self.ref)
            print(f"[woodelf_version] Installed woodelf@{self.ref}.")

        self._install_dir = install_dir
        return install_dir

    # ------------------------------------------------------------------
    # Subprocess runner
    # ------------------------------------------------------------------

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        tree_limit_depth: Optional[int],
        memory_crash_depth: Optional[int],
        feature_perturbation: str,
        use_interactions: bool,
    ) -> ApproachOutput:
        D, T = trained_model.params.D, trained_model.params.T

        if memory_crash_depth is not None and D >= memory_crash_depth:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)

        if (
            feature_perturbation == "interventional"
            and D >= 20
            and X_background is not None
        ):
            n = len(X_explain)
            m = len(X_background)
            if n * m > 10_000_000:
                return ApproachOutput(
                    elapsed_s=0.0,
                    not_supported=True,
                    estimation_description=(
                        f"Can not use the sparse approach with n={n} and m={m}, "
                        f"as the basic complexity is O(mn) which becomes too high here."
                    ),
                )

        install_dir = self._ensure_install()

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            model_path = tmp / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(trained_model.raw_model, f)

            x_explain_path = tmp / "x_explain.pkl"
            X_explain.to_pickle(x_explain_path)

            x_bg_path: Optional[Path] = None
            if X_background is not None:
                x_bg_path = tmp / "x_background.pkl"
                X_background.to_pickle(x_bg_path)

            args = {
                "model_pickle_path": str(model_path),
                "x_explain_path": str(x_explain_path),
                "x_background_path": str(x_bg_path) if x_bg_path else None,
                "D": D,
                "T": T,
                "tree_limit_depth": tree_limit_depth,
                "feature_perturbation": feature_perturbation,
                "use_interactions": use_interactions,
                "gpu": self.GPU,
            }

            env = os.environ.copy()
            env["PYTHONPATH"] = str(install_dir) + os.pathsep + env.get("PYTHONPATH", "")

            proc = subprocess.run(
                [sys.executable, str(_RUNNER)],
                input=json.dumps(args),
                capture_output=True,
                text=True,
                env=env,
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"woodelf runner failed (ref='{self.ref}'):\n{proc.stderr[-2000:]}"
            )

        _PREFIX = "TREEBRANCHMARKS_RESULT:"
        result_line = next(
            (ln[len(_PREFIX):] for ln in proc.stdout.splitlines() if ln.startswith(_PREFIX)),
            None,
        )
        if result_line is None:
            raise RuntimeError(
                f"woodelf runner produced no result line (ref='{self.ref}').\n"
                f"stdout: {proc.stdout!r}\n"
                f"stderr: {proc.stderr[-2000:]}"
            )
        out = json.loads(result_line)

        return ApproachOutput(
            elapsed_s=out["elapsed_s"],
            is_estimated=out.get("is_estimated", False),
            estimation_description=out.get("estimation_description", ""),
        )

    # ------------------------------------------------------------------
    # Task methods — each unpacks its per-task limits from self._limits
    # ------------------------------------------------------------------

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        lim, crash = self._limits.pd_shap
        return self._run(trained_model, X_explain, X_background,
                         lim, crash, "tree_path_dependent", False)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        lim, crash = self._limits.pd_iv
        return self._run(trained_model, X_explain, X_background,
                         lim, crash, "tree_path_dependent", True)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        lim, crash = self._limits.bg_shap
        return self._run(trained_model, X_explain, X_background,
                         lim, crash, "interventional", False)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        lim, crash = self._limits.bg_iv
        return self._run(trained_model, X_explain, X_background,
                         lim, crash, "interventional", True)
