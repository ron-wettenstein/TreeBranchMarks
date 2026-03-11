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
)
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_method import WoodelfApproach
from treebranchmarks.methods.woodelf_historical_methods import (
    WoodelfECAIApproach,
    WoodelfAAAIApproach,
    WoodelfHDApproach,
)
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
    # Approach classes
    "SHAPApproach",
    "WoodelfApproach", "WoodelfHDApproach",
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
]
