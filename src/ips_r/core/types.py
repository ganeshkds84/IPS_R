"""Provisional data contracts exchanged between components.

These are deliberately minimal, immutable containers. Research sprints may
extend them via ``metadata`` without breaking the interfaces.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ips_r.core.log import get_logger

logger = get_logger(__name__)


def _require_id(kind: str, value: str) -> None:
    """Validate that an identifier is a non-empty string."""
    if not isinstance(value, str) or not value:
        logger.debug("types.invalid_id", kind=kind, value=repr(value))
        raise ValueError(f"{kind}.id must be a non-empty string, got {value!r}")


def _require_finite(kind: str, name: str, value: float | None) -> None:
    """Validate that an optional numeric field is finite."""
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{kind}.{name} must be finite, got {value!r}")


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only shallow copy of ``mapping``."""
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class Query:
    """An information need submitted to the pipeline."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id("Query", self.id)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of retrievable content."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id("Document", self.id)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """A document paired with a relevance score (higher is more relevant)."""

    document: Document
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_finite("ScoredDocument", "score", self.score)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class Candidate:
    """An intermediate output proposed by a candidate generator."""

    id: str
    content: str
    score: float | None = None
    source_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id("Candidate", self.id)
        _require_finite("Candidate", "score", self.score)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class Decision:
    """The action chosen by a decision policy."""

    action: str
    confidence: float | None = None
    selected_ids: tuple[str, ...] = ()
    rationale: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("Decision.action must be a non-empty string")
        _require_finite("Decision", "confidence", self.confidence)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class Generation:
    """Final output produced for a query."""

    query_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id("Generation.query", self.query_id)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class Reference:
    """Gold-standard annotations for a query."""

    query_id: str
    answers: tuple[str, ...] = ()
    relevant_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id("Reference.query", self.query_id)
        object.__setattr__(self, "metadata", _frozen(self.metadata))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate (and optionally per-example) metric values."""

    metrics: Mapping[str, float]
    per_example: tuple[Mapping[str, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.metrics.items():
            _require_finite("EvaluationResult", name, value)
        object.__setattr__(self, "metrics", _frozen(self.metrics))
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "metrics": dict(self.metrics),
            "per_example": [dict(row) for row in self.per_example],
            "metadata": dict(self.metadata),
        }
