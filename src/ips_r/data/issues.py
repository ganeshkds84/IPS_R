"""Validation issues and structured validation reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class Severity(StrEnum):
    """How serious a validation issue is.

    ``ERROR`` records are unusable and dropped during preparation (when
    ``drop_invalid``); ``WARNING`` records are kept but suspicious; ``INFO`` is advisory.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueCode(StrEnum):
    """Machine-readable category of a validation issue."""

    DUPLICATE_ID = "duplicate_id"
    MISSING_FIELD = "missing_field"
    MALFORMED_RECORD = "malformed_record"
    INVALID_ENCODING = "invalid_encoding"
    EMPTY_TEXT = "empty_text"
    INVALID_METADATA = "invalid_metadata"
    CORRUPTED_PDF = "corrupted_pdf"
    INCONSISTENT_SCHEMA = "inconsistent_schema"
    DANGLING_REFERENCE = "dangling_reference"
    UNREADABLE_SOURCE = "unreadable_source"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single problem found in a dataset.

    Attributes:
        code: Issue category.
        severity: Issue severity.
        message: Human-readable description.
        record_id: Id of the affected document/question, when known.
        location: Where the problem was found (``file:line``, JSON path, ...).
    """

    code: IssueCode
    severity: Severity
    message: str
    record_id: str | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", IssueCode(self.code))
        object.__setattr__(self, "severity", Severity(self.severity))
        if not self.message:
            raise ValueError("ValidationIssue.message must be non-empty")

    @property
    def key(self) -> tuple[str, str, str | None, str | None]:
        """Identity used to de-duplicate repeated reports of the same issue."""
        return (self.code.value, self.message, self.record_id, self.location)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "record_id": self.record_id,
            "location": self.location,
        }


class IssueCollector:
    """Accumulates issues with de-duplication, full counts and a per-code example cap.

    Args:
        max_examples_per_code: Maximum issues *stored* per code; all issues are *counted*.
    """

    def __init__(self, max_examples_per_code: int = 100) -> None:
        if max_examples_per_code < 0:
            raise ValueError("max_examples_per_code must be >= 0")
        self._cap = max_examples_per_code
        self._seen: set[tuple[str, str, str | None, str | None]] = set()
        self._examples: list[ValidationIssue] = []
        self._by_code: Counter[str] = Counter()
        self._by_severity: Counter[str] = Counter()

    def add(self, issue: ValidationIssue) -> bool:
        """Record ``issue``. Returns ``False`` if an identical issue was already recorded."""
        if issue.key in self._seen:
            return False
        self._seen.add(issue.key)
        self._by_severity[issue.severity.value] += 1
        self._by_code[issue.code.value] += 1
        if self._by_code[issue.code.value] <= self._cap:
            self._examples.append(issue)
        return True

    def extend(self, issues: Iterable[ValidationIssue]) -> None:
        """Record several issues."""
        for issue in issues:
            self.add(issue)

    def clear(self) -> None:
        """Forget every recorded issue."""
        self._seen.clear()
        self._examples.clear()
        self._by_code.clear()
        self._by_severity.clear()

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Stored issues (capped per code), in insertion order."""
        return tuple(self._examples)

    @property
    def counts_by_code(self) -> dict[str, int]:
        """Total number of issues per code."""
        return dict(sorted(self._by_code.items()))

    @property
    def counts_by_severity(self) -> dict[str, int]:
        """Total number of issues per severity."""
        return {s.value: self._by_severity.get(s.value, 0) for s in Severity}

    @property
    def total(self) -> int:
        """Total number of distinct issues recorded."""
        return sum(self._by_code.values())

    def __len__(self) -> int:
        return self.total


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured outcome of validating a dataset.

    Attributes:
        dataset: Dataset name.
        num_documents_checked: Documents inspected.
        num_questions_checked: Questions inspected.
        counts_by_code: Total issues per :class:`IssueCode` value.
        counts_by_severity: Total issues per :class:`Severity` value.
        issues: Example issues (capped per code; see ``truncated``).
        truncated: Whether some issues were counted but not stored.
    """

    dataset: str
    num_documents_checked: int
    num_questions_checked: int
    counts_by_code: Mapping[str, int] = field(default_factory=dict)
    counts_by_severity: Mapping[str, int] = field(default_factory=dict)
    issues: tuple[ValidationIssue, ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts_by_code", MappingProxyType(dict(self.counts_by_code)))
        object.__setattr__(
            self, "counts_by_severity", MappingProxyType(dict(self.counts_by_severity))
        )
        object.__setattr__(self, "issues", tuple(self.issues))

    @classmethod
    def from_collector(
        cls,
        dataset: str,
        collector: IssueCollector,
        *,
        num_documents_checked: int,
        num_questions_checked: int,
    ) -> ValidationReport:
        """Build a report from an :class:`IssueCollector`."""
        return cls(
            dataset=dataset,
            num_documents_checked=num_documents_checked,
            num_questions_checked=num_questions_checked,
            counts_by_code=collector.counts_by_code,
            counts_by_severity=collector.counts_by_severity,
            issues=collector.issues,
            truncated=len(collector.issues) < collector.total,
        )

    @property
    def num_errors(self) -> int:
        """Number of error-severity issues."""
        return self.counts_by_severity.get(Severity.ERROR.value, 0)

    @property
    def num_warnings(self) -> int:
        """Number of warning-severity issues."""
        return self.counts_by_severity.get(Severity.WARNING.value, 0)

    @property
    def is_valid(self) -> bool:
        """``True`` when no error-severity issue was found."""
        return self.num_errors == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "dataset": self.dataset,
            "is_valid": self.is_valid,
            "num_documents_checked": self.num_documents_checked,
            "num_questions_checked": self.num_questions_checked,
            "num_errors": self.num_errors,
            "num_warnings": self.num_warnings,
            "counts_by_code": dict(self.counts_by_code),
            "counts_by_severity": dict(self.counts_by_severity),
            "truncated": self.truncated,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_markdown(self, *, max_rows: int = 50) -> str:
        """Render a human-readable Markdown summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"# Validation report: {self.dataset}",
            "",
            f"**Status:** {status} ({self.num_errors} errors, {self.num_warnings} warnings)",
            "",
            f"- Documents checked: {self.num_documents_checked}",
            f"- Questions checked: {self.num_questions_checked}",
            "",
            "## Issues by code",
            "",
        ]
        if self.counts_by_code:
            lines += ["| Code | Count |", "| --- | ---: |"]
            lines += [f"| `{code}` | {count} |" for code, count in self.counts_by_code.items()]
        else:
            lines.append("No issues found.")
        if self.issues:
            shown = self.issues[:max_rows]
            lines += [
                "",
                f"## Example issues ({len(shown)} of {sum(self.counts_by_code.values())})",
                "",
                "| Severity | Code | Record | Location | Message |",
                "| --- | --- | --- | --- | --- |",
            ]
            lines += [
                f"| {i.severity.value} | `{i.code.value}` | {_cell(i.record_id)} | "
                f"{_cell(i.location)} | {_cell(i.message)} |"
                for i in shown
            ]
        return "\n".join(lines) + "\n"


def _cell(value: str | None) -> str:
    """Escape a value for a Markdown table cell."""
    if value is None:
        return ""
    return value.replace("|", "\\|").replace("\n", " ")
