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

# Vectorized Linear TreeSHAP variants
WOODELF_VEC_SIMPLE = Method(
    name="woodelf_vec_simple",
    label="Woodelf Vec (Simple)",
    description="vectorized_linear_tree_shap with LinearTreeShapPathToMatricesSimple.",
)
WOODELF_VEC_SIMPLE_NLT = Method(
    name="woodelf_vec_simple_nlt",
    label="Woodelf Vec (Simple + NLT)",
    description="vectorized_linear_tree_shap with LinearTreeShapPathToMatricesSimple and neighbor-leaf trick.",
)
WOODELF_VEC_IMPROVED = Method(
    name="woodelf_vec_improved",
    label="Woodelf Vec (Improved)",
    description="vectorized_linear_tree_shap with LinearTreeShapPathToMatricesImproved.",
)
WOODELF_VEC_IMPROVED_NLT = Method(
    name="woodelf_vec_improved_nlt",
    label="Woodelf Vec (Improved + NLT)",
    description="vectorized_linear_tree_shap with LinearTreeShapPathToMatricesImproved and neighbor-leaf trick.",
)
WOODELF_VEC_DEFAULT = Method(
    name="woodelf_vec_default",
    label="Woodelf Vec (Default)",
    description="vectorized_linear_tree_shap with default p2m class.",
)
WOODELF_VEC_DEFAULT_NLT = Method(
    name="woodelf_vec_default_nlt",
    label="Woodelf Vec (Default + NLT)",
    description="vectorized_linear_tree_shap with default p2m class and neighbor-leaf trick.",
)

VECTORIZED_LINEAR_TREE_SHAP = Method(
    name="vectorized_linear_tree_shap",
    label="VectorizedLinearTreeSHAP",
    description="vectorized_linear_tree_shap.",
)