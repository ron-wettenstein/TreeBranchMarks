"""
Method: a named algorithmic method that can implement one or more tasks.

A Method is the primary unit of comparison in treebranchmarks.  Each Method
provides its own Approach implementation for every task it supports.

Canonical built-in instances (SHAP, WOODELF) live in
``treebranchmarks.methods.builtin``.  Custom methods are created by
instantiating Method directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Method:
    """
    Identifies and describes an algorithmic method.

    Parameters
    ----------
    name : str
        Machine-readable key used in JSON serialisation and scoring
        (e.g. ``"shap"``, ``"woodelf"``).
    label : str
        Human-readable display name used in charts and the HTML report
        (e.g. ``"SHAP"``, ``"Woodelf"``).
    description : str
        Optional free-text description shown in the report details panel.
    """

    name: str
    label: str
    description: str = ""

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Method):
            return self.name == other.name
        return NotImplemented

    def __repr__(self) -> str:
        return f"Method(name={self.name!r}, label={self.label!r})"

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description}
