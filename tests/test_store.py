"""Tests for stable_hash and CacheStore."""

import pytest
from treebranchmarks.cache.store import stable_hash


# ---------------------------------------------------------------------------
# stable_hash
# ---------------------------------------------------------------------------

def test_stable_hash_is_deterministic():
    obj = {"a": 1, "b": [1, 2, 3]}
    assert stable_hash(obj) == stable_hash(obj)


def test_stable_hash_dict_ordering_independent():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_different_values_differ():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_stable_hash_different_keys_differ():
    assert stable_hash({"a": 1}) != stable_hash({"b": 1})


def test_stable_hash_returns_hex_md5():
    h = stable_hash("hello")
    assert isinstance(h, str)
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h)


def test_stable_hash_non_serializable_uses_str():
    # default=str means non-serializable types don't raise
    from pathlib import Path
    h = stable_hash({"path": Path("/tmp/foo")})
    assert isinstance(h, str)


def test_stable_hash_nested_structures():
    obj = {"a": {"b": [1, 2, {"c": 3}]}}
    assert stable_hash(obj) == stable_hash(obj)


def test_stable_hash_empty_dict():
    h = stable_hash({})
    assert isinstance(h, str)
    assert len(h) == 32
