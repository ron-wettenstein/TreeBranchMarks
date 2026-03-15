from treebranchmarks.methods.builtin import (
    SHAP,
    WOODELF,
    WOODELF_VEC_SIMPLE,
    WOODELF_VEC_SIMPLE_NLT,
    WOODELF_VEC_IMPROVED,
    WOODELF_VEC_IMPROVED_NLT,
    WOODELF_VEC_DEFAULT,
    WOODELF_VEC_DEFAULT_NLT,
    WOODELF_VEC_RECURSIVE_NLT,
    VECTORIZED_LINEAR_TREE_SHAP,
    WOODELF_ECAI,
    WOODELF_AAAI,
    WOODELF_HD,
    LINEAR_TREESHAP_V6,
    TREEGRAD,
)
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_explainer_method import WoodelfApproach, WoodelfGPUApproach
from treebranchmarks.methods.woodelf_historical_methods import (
    WoodelfECAIApproach,
    WoodelfAAAIApproach,
    WoodelfHDHistoricalApproach,
)
from treebranchmarks.methods.linear_treeshap_v6_method import LinearTreeSHAPV6Approach
from treebranchmarks.methods.treegrad_method import TreeGradApproach
from treebranchmarks.methods.linear_tree_shap_method import (
    VectorizedLinearTreeSHAPApproach,
    VectorizedLinearTreeSHAPSimpleApproach,
    VectorizedLinearTreeSHAPSimpleNLTApproach,
    VectorizedLinearTreeSHAPImprovedApproach,
    VectorizedLinearTreeSHAPImprovedNLTApproach,
    VectorizedLinearTreeSHAPDefaultApproach,
    VectorizedLinearTreeSHAPDefaultNLTApproach,
    VectorizedLinearTreeSHAPRecursiveNLTApproach,
)

__all__ = [
    # Method constants
    "SHAP", "WOODELF",
    "WOODELF_VEC_SIMPLE", "WOODELF_VEC_SIMPLE_NLT",
    "WOODELF_VEC_IMPROVED", "WOODELF_VEC_IMPROVED_NLT",
    "WOODELF_VEC_DEFAULT", "WOODELF_VEC_DEFAULT_NLT",
    "WOODELF_VEC_RECURSIVE_NLT",
    "VECTORIZED_LINEAR_TREE_SHAP",
    "WOODELF_ECAI", "WOODELF_AAAI", "WOODELF_HD",
    "LINEAR_TREESHAP_V6", "TREEGRAD",
    # Approach classes
    "SHAPApproach",
    "WoodelfApproach", "WoodelfGPUApproach", "WoodelfHDHistoricalApproach",
    "WoodelfECAIApproach",
    "WoodelfAAAIApproach",
    "VectorizedLinearTreeSHAPApproach",
    "VectorizedLinearTreeSHAPSimpleApproach",
    "VectorizedLinearTreeSHAPSimpleNLTApproach",
    "VectorizedLinearTreeSHAPImprovedApproach",
    "VectorizedLinearTreeSHAPImprovedNLTApproach",
    "VectorizedLinearTreeSHAPDefaultApproach",
    "VectorizedLinearTreeSHAPDefaultNLTApproach",
    "VectorizedLinearTreeSHAPRecursiveNLTApproach",
    "LinearTreeSHAPV6Approach",
    "TreeGradApproach",
]
