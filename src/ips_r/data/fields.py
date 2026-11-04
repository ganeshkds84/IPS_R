"""Field validators shared by the dataset-layer models (raise :class:`ValueError`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


def require_str(model: str, name: str, value: Any, *, allow_empty: bool = True) -> None:
    """Validate that ``value`` is a string (optionally non-empty)."""
    if not isinstance(value, str):
        raise ValueError(f"{model}.{name} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ValueError(f"{model}.{name} must be a non-empty string")


def require_optional_str(model: str, name: str, value: Any) -> None:
    """Validate that ``value`` is ``None`` or a string."""
    if value is not None:
        require_str(model, name, value)


def require_str_tuple(model: str, name: str, value: Any) -> tuple[str, ...]:
    """Validate and normalise a sequence of strings into a tuple."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{model}.{name} must be a sequence of strings, got {value!r}")
    items = tuple(value)
    for item in items:
        require_str(model, name, item)
    return items


def require_optional_int(model: str, name: str, value: Any, *, minimum: int) -> None:
    """Validate that ``value`` is ``None`` or an integer ``>= minimum``."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{model}.{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{model}.{name} must be >= {minimum}, got {value}")


def frozen_mapping(model: str, name: str, value: Any) -> Mapping[str, Any]:
    """Return a read-only copy of a mapping field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{model}.{name} must be a mapping, got {type(value).__name__}")
    return MappingProxyType(dict(value))


def set_field(instance: object, name: str, value: Any) -> None:
    """Assign a field of a frozen dataclass during ``__post_init__``."""
    object.__setattr__(instance, name, value)
