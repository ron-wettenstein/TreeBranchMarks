"""Tests for TreeParameters and partial_tree_params."""

import pytest
from treebranchmarks.core.params import TreeParameters, EnsembleType, partial_tree_params


def _params(**kwargs) -> TreeParameters:
    defaults = dict(T=10, D=6, L=32.0, F=20, ensemble_type=EnsembleType.LIGHTGBM, n=100, m=50)
    return TreeParameters(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_basic_construction():
    p = _params()
    assert p.T == 10
    assert p.D == 6
    assert p.L == 32.0
    assert p.F == 20
    assert p.n == 100
    assert p.m == 50
    assert p.ensemble_type == EnsembleType.LIGHTGBM


def test_partial_tree_params_zero_n_m():
    p = partial_tree_params(T=5, D=3, L=8.0, F=10, ensemble_type=EnsembleType.XGBOOST)
    assert p.n == 0
    assert p.m == 0
    assert p.T == 5


# ---------------------------------------------------------------------------
# with_run_params
# ---------------------------------------------------------------------------

def test_with_run_params_fills_n_m():
    base = partial_tree_params(T=5, D=3, L=8.0, F=10, ensemble_type=EnsembleType.XGBOOST)
    full = base.with_run_params(n=1000, m=200)
    assert full.n == 1000
    assert full.m == 200
    assert full.T == 5  # unchanged


def test_with_run_params_returns_new_instance():
    base = _params(n=0, m=0)
    full = base.with_run_params(n=500, m=100)
    assert base.n == 0  # original unchanged
    assert full.n == 500


def test_with_run_params_default_m_zero():
    base = partial_tree_params(T=1, D=1, L=1.0, F=1, ensemble_type=EnsembleType.RANDOM_FOREST)
    full = base.with_run_params(n=10)
    assert full.m == 0


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

def test_cache_key_is_deterministic():
    p = _params()
    assert p.cache_key() == p.cache_key()


def test_cache_key_same_params_same_key():
    p1 = _params(n=100, m=50)
    p2 = _params(n=100, m=50)
    assert p1.cache_key() == p2.cache_key()


def test_cache_key_differs_on_n():
    p1 = _params(n=100)
    p2 = _params(n=200)
    assert p1.cache_key() != p2.cache_key()


def test_cache_key_differs_on_depth():
    p1 = _params(D=6)
    p2 = _params(D=12)
    assert p1.cache_key() != p2.cache_key()


def test_cache_key_differs_on_ensemble_type():
    p1 = _params(ensemble_type=EnsembleType.LIGHTGBM)
    p2 = _params(ensemble_type=EnsembleType.XGBOOST)
    assert p1.cache_key() != p2.cache_key()


def test_cache_key_is_hex_string():
    key = _params().cache_key()
    assert isinstance(key, str)
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------

def test_as_dict_contains_all_fields():
    p = _params()
    d = p.as_dict()
    assert d["T"] == 10
    assert d["D"] == 6
    assert d["L"] == 32.0
    assert d["F"] == 20
    assert d["n"] == 100
    assert d["m"] == 50
    assert d["ensemble_type"] == "lightgbm"  # serialized as string value


def test_as_dict_ensemble_type_is_string():
    p = _params(ensemble_type=EnsembleType.XGBOOST)
    assert p.as_dict()["ensemble_type"] == "xgboost"


# ---------------------------------------------------------------------------
# Immutability (frozen dataclass)
# ---------------------------------------------------------------------------

def test_frozen():
    p = _params()
    with pytest.raises((AttributeError, TypeError)):
        p.n = 999  # type: ignore[misc]
