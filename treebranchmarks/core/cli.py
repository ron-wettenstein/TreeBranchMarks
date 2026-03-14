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
    Extra path to write the results JSON.  The JSON is written to both the
    normal ``results/{name}.json`` **and** PATH after every completed
    mission, so partial results are preserved if the run is interrupted.
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
            "Extra path to write the results JSON "
            "(dual-write after every mission)."
        ),
    )
    args = parser.parse_args()

    experiment = build_fn()

    if args.methods:
        experiment.method_filter = [m.lower() for m in args.methods]
    if args.result_location is not None:
        experiment.extra_result_paths = [args.result_location]

    experiment.run()
    report_path = experiment.generate_html()
    print(f"\nOpen the report in your browser:\n  {report_path.resolve()}")
