"""Tests for MethodResultCache group key generation."""

from treebranchmarks.cache.method_cache import _group_key
from treebranchmarks.core.params import TreeParameters, EnsembleType


def _params(**kwargs) -> TreeParameters:
    defaults = dict(T=10, D=6, L=32.0, F=20, ensemble_type=EnsembleType.LIGHTGBM, n=100, m=50)
    return TreeParameters(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# _group_key determinism and stability
# ---------------------------------------------------------------------------

def test_group_key_is_deterministic():
    p = _params()
    k1 = _group_key("shap", "mission_a", "task_a", p)
    k2 = _group_key("shap", "mission_a", "task_a", p)
    assert k1 == k2


def test_group_key_same_inputs_same_key():
    p1 = _params(n=100, m=50, D=6)
    p2 = _params(n=100, m=50, D=6)
    assert _group_key("shap", "m", "t", p1) == _group_key("shap", "m", "t", p2)


def test_group_key_is_hex_string():
    k = _group_key("shap", "mission", "task", _params())
    assert isinstance(k, str)
    assert len(k) == 32
    assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# Key changes when any input changes
# ---------------------------------------------------------------------------

def test_group_key_differs_on_approach_name():
    p = _params()
    assert _group_key("shap", "m", "t", p) != _group_key("woodelf", "m", "t", p)


def test_group_key_differs_on_mission_name():
    p = _params()
    assert _group_key("shap", "mission_a", "t", p) != _group_key("shap", "mission_b", "t", p)


def test_group_key_differs_on_task_name():
    p = _params()
    assert _group_key("shap", "m", "task_a", p) != _group_key("shap", "m", "task_b", p)


def test_group_key_differs_on_n():
    assert _group_key("shap", "m", "t", _params(n=100)) != _group_key("shap", "m", "t", _params(n=200))


def test_group_key_differs_on_m():
    assert _group_key("shap", "m", "t", _params(m=50)) != _group_key("shap", "m", "t", _params(m=100))


def test_group_key_differs_on_depth():
    assert _group_key("shap", "m", "t", _params(D=6)) != _group_key("shap", "m", "t", _params(D=12))


def test_group_key_differs_on_ensemble_type():
    p_lgbm = _params(ensemble_type=EnsembleType.LIGHTGBM)
    p_xgb  = _params(ensemble_type=EnsembleType.XGBOOST)
    assert _group_key("shap", "m", "t", p_lgbm) != _group_key("shap", "m", "t", p_xgb)
