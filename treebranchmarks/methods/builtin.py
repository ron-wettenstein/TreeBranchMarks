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

WOODELF_ECAI = Method(
    name="woodelf_ecai",
    label="Woodelf ECAI",
    description="Woodelf ECAI algorithm (WDNF-based path-dependent/background SHAP).",
)

WOODELF_AAAI = Method(
    name="woodelf_aaai",
    label="Woodelf AAAI",
    description="Woodelf AAAI algorithm (cube-based path-dependent/background SHAP).",
)

ORIGINAL_WOODELF = Method(
    name="original_woodelf",
    label="OriginalWoodelf",
    description="Woodelf original algorithm (woodelf.simple_woodelf cube-based SHAP).",
)

ORIGINAL_WOODELF_GPU = Method(
    name="original_woodelf_gpu",
    label="OriginalWoodelf GPU",
    description="Woodelf original algorithm (woodelf.simple_woodelf cube-based SHAP) accelerated on GPU (CuPy).",
)

WOODELF_HD = Method(
    name="woodelf_hd",
    label="WoodelfHD",
    description="Woodelf high-depth algorithm (woodelf_for_high_depth).",
)

WOODELF_HD_GPU = Method(
    name="woodelf_hd_gpu",
    label="WoodelfHD GPU",
    description="Woodelf high-depth algorithm (woodelf_for_high_depth) accelerated on GPU (CuPy).",
)

WOODELF_VEC_RECURSIVE_NLT = Method(
    name="woodelf_vec_recursive_nlt",
    label="Woodelf Vec (Recursive + NLT)",
    description="vectorized_linear_tree_shap with LinearTreeShapPathToMatricesSimpleNeighborTrickAbstract + improved_linear_tree_shap_magic.",
)

WOODELF_VEC_V6 = Method(
    name="woodelf_vec_v6",
    label="Woodelf Vec V6",
    description="vectorized_linear_tree_shap with LinearTreeShapV6PathToMatrices.",
)

WOODELF_VEC_V6_SIMPLE = Method(
    name="woodelf_vec_v6_simple",
    label="Woodelf Vec V6 Simple",
    description="vectorized_linear_tree_shap with LinearTreeShapPathToMatricesV6Simple (SimpleNeighborTrickAbstract + linear_tree_shap_v6).",
)

LINEAR_TREESHAP_V6 = Method(
    name="linear_treeshap_v6",
    label="Linear TreeSHAP V6",
    description="Path-dependent SHAP via telescoping + Gauss-Legendre quadrature (woodelf tree API).",
)

TREEGRAD = Method(
    name="treegrad",
    label="TreeGrad",
    description="Path-dependent SHAP via TreeGrad (https://github.com/watml/TreeGrad). sklearn models only.",
)

PLTREESHAP_FASTTREESHAP = Method(
    name="pltreeshap_fasttreeshap",
    label="PLTreeSHAP + FastTreeSHAP",
    description="PLTreeSHAP (pltreeshap.PLTreeExplainer) for background tasks; FastTreeSHAP (fasttreeshap.TreeExplainer) for path-dependent tasks.",
)