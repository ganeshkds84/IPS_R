"""Validation of raw records (schema checks) and canonical records (content checks).

Two layers cooperate:

- :class:`RecordSchema` / :class:`SchemaTracker` check *raw* source records inside
  adapters: missing fields, wrong types and records whose shape deviates from the rest
  of the file.
- :class:`DatasetValidator` checks *canonical* :class:`~ips_r.data.models.Document` and
  :class:`~ips_r.data.models.Question` objects as they stream past: duplicate ids,
  empty text, encoding defects, implausible metadata and dangling references.

Both report :class:`~ips_r.data.issues.ValidationIssue` objects rather than raising, so
a single bad record never aborts processing of a large dataset.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import NoneType
from typing import Any, Final

from ips_r.config.schema import ValidationConfig
from ips_r.core.log import get_logger
from ips_r.data.issues import IssueCode, IssueCollector, Severity, ValidationIssue, ValidationReport
from ips_r.data.models import Document, DocumentMetadata, Question
from ips_r.data.text import encoding_problems

logger = get_logger(__name__)

DOI_PATTERN: Final = re.compile(r"^10\.\d{4,9}/\S+$")
LANGUAGE_PATTERN: Final = re.compile(r"^(?:[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")
URL_PATTERN: Final = re.compile(r"^[a-z][a-z0-9+.-]*://\S+$", re.IGNORECASE)


_JSON_TYPE_NAMES: Final[tuple[tuple[Any, str], ...]] = (
    (NoneType, "null"),
    (bool, "bool"),  # before int: bool is a subclass of int
    (int | float, "number"),
    (str, "string"),
    (list | tuple, "array"),
    (Mapping, "object"),
)


def _type_name(value: Any) -> str:
    """Return a short JSON-ish type name for ``value``."""
    for types, name in _JSON_TYPE_NAMES:
        if isinstance(value, types):
            return name
    return type(value).__name__


@dataclass(frozen=True, slots=True)
class RecordSchema:
    """Expected shape of a raw JSON-like record.

    Attributes:
        name: Schema name used in messages (e.g. ``"squad.question"``).
        required: Field name → accepted JSON type names (``string``, ``number``, ...).
        optional: Optional fields with their accepted type names.
    """

    name: str
    required: Mapping[str, frozenset[str]]
    optional: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def check(
        self, record: Any, *, record_id: str | None = None, location: str | None = None
    ) -> list[ValidationIssue]:
        """Return error issues if ``record`` lacks required fields or has wrong types."""
        if not isinstance(record, Mapping):
            return [
                ValidationIssue(
                    IssueCode.MALFORMED_RECORD,
                    Severity.ERROR,
                    f"{self.name}: expected an object, got {_type_name(record)}",
                    record_id,
                    location,
                )
            ]
        issues: list[ValidationIssue] = []
        for key, accepted in self.required.items():
            if key not in record or record[key] is None:
                issues.append(
                    ValidationIssue(
                        IssueCode.MISSING_FIELD,
                        Severity.ERROR,
                        f"{self.name}: missing required field '{key}'",
                        record_id,
                        location,
                    )
                )
            elif _type_name(record[key]) not in accepted:
                issues.append(self._type_issue(key, record[key], accepted, record_id, location))
        for key, accepted in self.optional.items():
            if record.get(key) is not None and _type_name(record[key]) not in accepted:
                issues.append(self._type_issue(key, record[key], accepted, record_id, location))
        return issues

    def _type_issue(
        self,
        key: str,
        value: Any,
        accepted: frozenset[str],
        record_id: str | None,
        location: str | None,
    ) -> ValidationIssue:
        """Build the issue reported for a field of the wrong type."""
        return ValidationIssue(
            IssueCode.MALFORMED_RECORD,
            Severity.ERROR,
            f"{self.name}: field '{key}' should be {'/'.join(sorted(accepted))}, "
            f"got {_type_name(value)}",
            record_id,
            location,
        )


class SchemaTracker:
    """Detects records whose shape differs from the first record of a source.

    The first observed record defines the reference *signature* (field name → type).
    Each distinct deviating signature is reported once as an ``inconsistent_schema``
    warning, so heterogeneous files are flagged without flooding the report.

    Args:
        name: Source name used in messages.
        ignore: Fields excluded from the signature (e.g. legitimately optional ones).
    """

    def __init__(self, name: str, *, ignore: Iterable[str] = ()) -> None:
        self._name = name
        self._ignore = frozenset(ignore)
        self._reference: frozenset[tuple[str, str]] | None = None
        self._reported: set[frozenset[tuple[str, str]]] = set()

    def observe(
        self, record: Mapping[str, Any], *, location: str | None = None
    ) -> list[ValidationIssue]:
        """Compare ``record`` to the reference signature; return at most one new warning."""
        signature = frozenset(
            (key, _type_name(value)) for key, value in record.items() if key not in self._ignore
        )
        if self._reference is None:
            self._reference = signature
            return []
        if signature == self._reference or signature in self._reported:
            return []
        self._reported.add(signature)
        ref_keys = {key for key, _ in self._reference}
        keys = {key for key, _ in signature}
        details = []
        if missing := sorted(ref_keys - keys):
            details.append(f"missing {missing}")
        if extra := sorted(keys - ref_keys):
            details.append(f"unexpected {extra}")
        if changed := sorted(
            k for k, t in signature if k in ref_keys and (k, t) not in self._reference
        ):
            details.append(f"type changed for {changed}")
        return [
            ValidationIssue(
                IssueCode.INCONSISTENT_SCHEMA,
                Severity.WARNING,
                f"{self._name}: record shape differs from the first record ({'; '.join(details)})",
                location=location,
            )
        ]

    def observe_columns(
        self, num_columns: int, *, location: str | None = None
    ) -> list[ValidationIssue]:
        """Column-count variant of :meth:`observe` for delimited files."""
        return self.observe({f"column_{i}": "" for i in range(num_columns)}, location=location)


class DatasetValidator:
    """Streaming content validator for canonical documents and questions.

    Call :meth:`check_document` / :meth:`check_question` for every record, then
    :meth:`report` once. The validator keeps only ids in memory.

    Args:
        settings: Validation rules; defaults match ``configs/preparation/default.yaml``.
    """

    def __init__(self, settings: ValidationConfig | None = None) -> None:
        self._settings = settings or ValidationConfig()
        if self._settings.min_year > self._settings.max_year:
            raise ValueError("validation.min_year must be <= validation.max_year")
        self._collector = IssueCollector(self._settings.max_issues_per_code)
        self._document_ids: set[str] = set()
        self._question_ids: set[str] = set()
        self._pending_refs: dict[str, set[str]] = {}
        self._num_documents = 0
        self._num_questions = 0

    @property
    def settings(self) -> ValidationConfig:
        """The rules in effect."""
        return self._settings

    def check_document(self, document: Document) -> list[ValidationIssue]:
        """Validate one document and return the issues found for it."""
        self._num_documents += 1
        issues: list[ValidationIssue] = []
        if document.id in self._document_ids:
            issues.append(
                self._issue(
                    IssueCode.DUPLICATE_ID, Severity.ERROR, "duplicate document id", document.id
                )
            )
        self._document_ids.add(document.id)
        self._pending_refs.pop(document.id, None)
        if not document.text.strip():
            issues.append(
                self._issue(
                    IssueCode.EMPTY_TEXT, Severity.ERROR, "document text is empty", document.id
                )
            )
        if self._settings.check_encoding:
            issues += [
                self._issue(
                    IssueCode.INVALID_ENCODING, Severity.WARNING, f"document text {p}", document.id
                )
                for p in encoding_problems(document.text)
            ]
        if self._settings.check_metadata:
            issues += self._check_metadata(document.id, document.metadata)
        self._collector.extend(issues)
        return issues

    def check_question(self, question: Question) -> list[ValidationIssue]:
        """Validate one question and return the issues found for it."""
        self._num_questions += 1
        issues: list[ValidationIssue] = []
        if question.id in self._question_ids:
            issues.append(
                self._issue(
                    IssueCode.DUPLICATE_ID, Severity.ERROR, "duplicate question id", question.id
                )
            )
        self._question_ids.add(question.id)
        if not question.text.strip():
            issues.append(
                self._issue(
                    IssueCode.EMPTY_TEXT, Severity.ERROR, "question text is empty", question.id
                )
            )
        if self._settings.check_encoding:
            issues += [
                self._issue(
                    IssueCode.INVALID_ENCODING, Severity.WARNING, f"question text {p}", question.id
                )
                for p in encoding_problems(question.text)
            ]
        if any(not answer.strip() for answer in question.answers):
            issues.append(
                self._issue(
                    IssueCode.EMPTY_TEXT,
                    Severity.WARNING,
                    "question has an empty answer",
                    question.id,
                )
            )
        for doc_id in question.supporting_document_ids:
            if doc_id not in self._document_ids:
                self._pending_refs.setdefault(doc_id, set()).add(question.id)
        self._collector.extend(issues)
        return issues

    def report(
        self, dataset: str, ingestion_issues: Iterable[ValidationIssue] = ()
    ) -> ValidationReport:
        """Finalise validation and return the report.

        Supporting-document ids that never matched a document are reported as
        ``dangling_reference`` warnings. ``ingestion_issues`` (from the adapter) are merged in.
        """
        self._collector.extend(ingestion_issues)
        for doc_id in sorted(self._pending_refs):
            for question_id in sorted(self._pending_refs[doc_id]):
                self._collector.add(
                    self._issue(
                        IssueCode.DANGLING_REFERENCE,
                        Severity.WARNING,
                        f"supporting document '{doc_id}' not found",
                        question_id,
                    )
                )
        self._pending_refs.clear()
        report = ValidationReport.from_collector(
            dataset,
            self._collector,
            num_documents_checked=self._num_documents,
            num_questions_checked=self._num_questions,
        )
        logger.info(
            "validation.completed",
            dataset=dataset,
            errors=report.num_errors,
            warnings=report.num_warnings,
            documents=report.num_documents_checked,
            questions=report.num_questions_checked,
        )
        return report

    def _check_metadata(self, record_id: str, metadata: DocumentMetadata) -> list[ValidationIssue]:
        """Return ``invalid_metadata`` warnings for implausible bibliographic fields."""
        problems: list[str] = []
        if metadata.year is not None and not (
            self._settings.min_year <= metadata.year <= self._settings.max_year
        ):
            problems.append(
                f"year {metadata.year} outside [{self._settings.min_year}, {self._settings.max_year}]"
            )
        if metadata.doi is not None and not DOI_PATTERN.match(metadata.doi):
            problems.append(f"malformed DOI {metadata.doi!r}")
        if metadata.language is not None and not LANGUAGE_PATTERN.match(metadata.language):
            problems.append(f"invalid language code {metadata.language!r}")
        if metadata.url is not None and not URL_PATTERN.match(metadata.url):
            problems.append(f"invalid URL {metadata.url!r}")
        if metadata.title is not None and not metadata.title.strip():
            problems.append("blank title")
        if any(not author.strip() for author in metadata.authors):
            problems.append("blank author name")
        if any(not keyword.strip() for keyword in metadata.keywords):
            problems.append("blank keyword")
        return [
            self._issue(IssueCode.INVALID_METADATA, Severity.WARNING, problem, record_id)
            for problem in problems
        ]

    @staticmethod
    def _issue(
        code: IssueCode, severity: Severity, message: str, record_id: str
    ) -> ValidationIssue:
        """Build an issue attached to ``record_id``."""
        return ValidationIssue(code, severity, message, record_id)


def has_errors(issues: Iterable[ValidationIssue]) -> bool:
    """Return whether any issue in ``issues`` is error-severity."""
    return any(issue.severity is Severity.ERROR for issue in issues)
