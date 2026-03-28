"""Tests for the Method dataclass."""

from treebranchmarks.core.method import Method


def _method(name="shap", label="SHAP", description="") -> Method:
    return Method(name=name, label=label, description=description)


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------

def test_equality_by_name():
    assert _method(name="shap") == _method(name="shap", label="Different Label")


def test_inequality_different_names():
    assert _method(name="shap") != _method(name="woodelf")


def test_hash_equal_for_same_name():
    assert hash(_method(name="shap")) == hash(_method(name="shap", label="X"))


def test_hash_differs_for_different_names():
    assert hash(_method(name="shap")) != hash(_method(name="woodelf"))


def test_usable_as_dict_key():
    d = {_method(name="shap"): 1, _method(name="woodelf"): 2}
    assert d[_method(name="shap")] == 1
    assert d[_method(name="woodelf")] == 2


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------

def test_as_dict_fields():
    m = _method(name="shap", label="SHAP", description="Reference implementation.")
    d = m.as_dict()
    assert d == {"name": "shap", "label": "SHAP", "description": "Reference implementation."}


def test_as_dict_empty_description():
    d = _method(name="x", label="X").as_dict()
    assert d["description"] == ""
