"""
CLI runner for treebranchmarks experiments.

Usage in an experiment's __main__:

    from treebranchmarks.core.cli import run_experiment_cli
    from my_experiment import build_experiment

    if __name__ == "__main__":
        run_experiment_cli(build_experiment)

CLI flags
---------
--method METHOD
    Only run approaches whose ``method.name`` matches METHOD
    (case-insensitive).  Repeatable: ``--method Woodelf --method SHAP``.
    If omitted, all methods run.

--result_location PATH
    Extra path to write the method cache file.  Written after every
    approach result, so partial progress is preserved if the run is
    interrupted.  To resume, copy this file to
    ``cache/method_results/{experiment_name}/{method_name}.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


def run_experiment_cli(build_fn: Callable) -> None:
    """
    Parse CLI args, configure and run an experiment.

    Parameters
    ----------
    build_fn : Callable
        Zero-argument function that returns a fully-configured ``Experiment``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        metavar="METHOD",
        help=(
            "Method name to run (case-insensitive match against "
            "approach.method.name).  Repeatable.  "
            "If omitted, all methods are run."
        ),
    )
    parser.add_argument(
        "--result_location",
        type=Path,
        default=None,
        help=(
            "Extra path to write the method cache file "
            "(dual-write after every approach result)."
        ),
    )
    args = parser.parse_args()

    experiment = build_fn()

    if args.methods:
        experiment.method_filter = [m.lower() for m in args.methods]
    if args.result_location is not None:
        experiment.extra_method_cache_paths = [args.result_location]

    experiment.run()
