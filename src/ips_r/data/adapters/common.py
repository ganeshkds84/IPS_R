"""Helpers shared by dataset adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

STR: Final = frozenset({"string"})
"""Accepted JSON types for string fields."""

ID: Final = frozenset({"string", "number"})
"""Accepted JSON types for identifier fields (some sources use integers)."""


def unique(values: Iterable[str]) -> tuple[str, ...]:
    """De-duplicate ``values`` preserving first-seen order."""
    return tuple(dict.fromkeys(values))


def to_title(raw: str) -> str:
    """Turn a Wikipedia-style title (``Super_Bowl_50``) into display form."""
    return raw.replace("_", " ").strip()
