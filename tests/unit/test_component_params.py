"""Tests for typed parameter helpers and guarded iteration on AbstractComponent."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ips_r.core.exceptions import ComponentError, ConfigurationError
from tests.fakes import FakeRetriever


def _component(**params: Any) -> FakeRetriever:
    return FakeRetriever(params, name="c")


def test_typed_params_accept_valid_values() -> None:
    c = _component(i=3, f=0.5, b=True, s="x", l=["a", "b"], n=None)
    assert c.int_param("i", minimum=1) == 3
    assert c.optional_int_param("n") is None
    assert c.optional_int_param("i") == 3
    assert c.float_param("f", minimum=0, maximum=1) == 0.5
    assert c.float_param("i") == 3.0
    assert c.bool_param("b") is True
    assert c.str_param("s") == "x"
    assert c.optional_str_param("n") is None
    assert c.optional_str_param("s") == "x"
    assert c.str_list_param("l") == ("a", "b")
    assert c.int_param("missing", 7) == 7


@pytest.mark.parametrize(
    ("method", "key", "value", "kwargs"),
    [
        ("int_param", "k", "3", {}),
        ("int_param", "k", True, {}),
        ("int_param", "k", 0, {"minimum": 1}),
        ("float_param", "k", "x", {}),
        ("float_param", "k", 2.0, {"maximum": 1.0}),
        ("bool_param", "k", 1, {}),
        ("str_param", "k", "", {}),
        ("str_list_param", "k", "abc", {}),
        ("str_list_param", "k", [1], {}),
    ],
)
def test_typed_params_reject_invalid_values(
    method: str, key: str, value: Any, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ConfigurationError, match="parameter 'k'"):
        getattr(_component(**{key: value}), method)(key, **kwargs)


def test_missing_required_param() -> None:
    with pytest.raises(ConfigurationError, match="requires"):
        _component().int_param("absent")


def test_guarded_iter_limits_wraps_and_closes() -> None:
    c = _component()
    closed: list[bool] = []

    def numbers() -> Iterator[int]:
        try:
            yield from range(10)
        finally:
            closed.append(True)

    assert list(c._guarded_iter("op", numbers, limit=3)) == [0, 1, 2]
    assert closed == [True]
    assert c.is_ready

    def failing() -> Iterator[int]:
        yield 1
        raise RuntimeError("boom")

    with pytest.raises(ComponentError, match="op failed: boom"):
        list(c._guarded_iter("op", failing))
