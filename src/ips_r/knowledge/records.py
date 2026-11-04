"""Flat JSON-compatible records of frozen dataclasses (knowledge objects).

:func:`to_record` encodes enums as their values, nested dataclasses (``SourceSpan``) as
objects and tuples as lists; :func:`from_record` inverts it, guided by the class's
resolved type hints (:func:`field_types`).
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from enum import StrEnum
from types import NoneType, UnionType
from typing import Any, Final

_VARIADIC_TUPLE_ARGS: Final = 2  # tuple[X, ...]
_HINTS: dict[type[Any], dict[str, Any]] = {}


def _encode(value: Any) -> Any:
    """Convert a field value to a JSON-compatible structure."""
    if isinstance(value, StrEnum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_encode(v) for v in value]
    return value


def to_record(obj: Any) -> dict[str, Any]:
    """Return the JSON-compatible record of a knowledge object."""
    return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


def field_types(cls: type[Any]) -> dict[str, Any]:
    """Resolved type hints of a knowledge object class (cached)."""
    if cls not in _HINTS:
        _HINTS[cls] = typing.get_type_hints(cls)
    return _HINTS[cls]


def strip_optional(hint: Any) -> Any:
    """Return ``X`` for ``X | None``."""
    if isinstance(hint, UnionType) or typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not NoneType]
        if len(args) == 1:
            return args[0]
    return hint


def _decode(hint: Any, value: Any) -> Any:
    """Inverse of :func:`_encode` guided by the field's type hint."""
    if value is None:
        return None
    hint = strip_optional(hint)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return hint(**{k: _decode(v, value[k]) for k, v in field_types(hint).items() if k in value})
    if isinstance(hint, type) and issubclass(hint, StrEnum):
        return hint(value)
    if typing.get_origin(hint) is tuple:
        args = typing.get_args(hint)
        inner = args[0] if args else Any
        if len(args) == _VARIADIC_TUPLE_ARGS and args[1] is Ellipsis:
            return tuple(_decode(inner, v) for v in value)
        return tuple(_decode(a, v) for a, v in zip(args, value, strict=True))
    return value


def from_record[T](cls: type[T], record: Mapping[str, Any]) -> T:
    """Rebuild a knowledge object from :func:`to_record` output.

    Raises:
        ValueError: If the record violates the object's invariants.
        KeyError: If a required field is missing.
    """
    hints = field_types(cls)
    kwargs = {name: _decode(hint, record[name]) for name, hint in hints.items() if name in record}
    return cls(**kwargs)
