"""
Built-in Method instances shipped with treebranchmarks.

Import these when defining Approach subclasses or constructing tasks:

    from treebranchmarks.methods import SHAP, WOODELF
"""

from treebranchmarks.core.method import Method

SHAP = Method(
    name="shap",
    label="SHAP",
    description="Reference implementation from the shap library (shap.TreeExplainer).",
)

WOODELF = Method(
    name="woodelf",
    label="Woodelf",
    description="Woodelf TreeExplainer implementation.",
)
