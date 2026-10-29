"""Abstract dataset: streams canonical documents and questions from a source.

Implementations override two generator hooks, :meth:`AbstractDataset._iter_documents`
and :meth:`AbstractDataset._iter_questions`, that convert source records into
:mod:`ips_r.data.models` objects. Problems in source records are reported via
:meth:`AbstractDataset._report` instead of raising, so one bad record never aborts
ingestion. The base class adds lazy setup, error wrapping, ``max_*`` limits, caching
and conversion to the retrieval contracts in :mod:`ips_r.core.types`.

Common parameters (``params`` of ``configs/dataset/*.yaml``):

``split``           informational split label (``train``, ``dev``, ...)
``max_documents``   stop after this many documents (``null`` = all)
``max_questions``   stop after this many questions (``null`` = all)
``strict``          raise on the first error-severity ingestion issue
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core import types as core
from ips_r.core.exceptions import ComponentError, ConfigurationError
from ips_r.core.registry import Registry
from ips_r.data.issues import IssueCode, IssueCollector, Severity, ValidationIssue
from ips_r.data.models import DatasetMetadata, Document, Question

UNLIMITED_ISSUES = 1_000_000
"""Example cap for ingestion issues; counts are always complete."""


class AbstractDataset(AbstractComponent):
    """Uniform interface over every dataset source.

    Class attributes describe the dataset for :meth:`describe`; implementations
    override them.
    """

    component_type: ClassVar[str] = "dataset"
    description: ClassVar[str] = ""
    homepage: ClassVar[str | None] = None
    license: ClassVar[str | None] = None
    language: ClassVar[str | None] = None

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._split = self.optional_str_param("split")
        self._max_documents = self.optional_int_param("max_documents", minimum=1)
        self._max_questions = self.optional_int_param("max_questions", minimum=1)
        self._strict = self.bool_param("strict", False)
        self._issues = IssueCollector(UNLIMITED_ISSUES)
        self._documents_cache: list[Document] | None = None
        self._questions_cache: list[Question] | None = None

    # ------------------------------------------------------------------ #
    # Canonical API
    # ------------------------------------------------------------------ #
    def describe(self) -> DatasetMetadata:
        """Return static provenance information (counts are filled in by preparation)."""
        return DatasetMetadata(
            name=self.name,
            adapter=type(self).__name__,
            description=self.description,
            split=self._split,
            version=self.optional_str_param("version"),
            homepage=self.homepage,
            license=self.license,
            language=self.language,
            sources=tuple(str(path) for path in self._sources()),
        )

    def iter_documents(self) -> Iterator[Document]:
        """Stream documents (honouring ``max_documents``).

        Raises:
            ComponentError: If reading fails unexpectedly, or on the first error-severity
                ingestion issue when ``strict`` is enabled.
        """
        return self._guarded_iter("documents", self._iter_documents, limit=self._max_documents)

    def iter_questions(self) -> Iterator[Question]:
        """Stream questions (honouring ``max_questions``).

        Raises:
            ComponentError: As for :meth:`iter_documents`.
        """
        return self._guarded_iter("questions", self._iter_questions, limit=self._max_questions)

    def load_documents(self) -> list[Document]:
        """Return all documents, cached until :meth:`teardown`."""
        if self._documents_cache is None:
            self._documents_cache = list(self.iter_documents())
            self._logger.info("dataset.documents.loaded", count=len(self._documents_cache))
        return self._documents_cache

    def load_questions(self) -> list[Question]:
        """Return all questions, cached until :meth:`teardown`."""
        if self._questions_cache is None:
            self._questions_cache = list(self.iter_questions())
            self._logger.info("dataset.questions.loaded", count=len(self._questions_cache))
        return self._questions_cache

    def ingestion_issues(self) -> tuple[ValidationIssue, ...]:
        """Issues found in source records so far (de-duplicated)."""
        return self._issues.issues

    # ------------------------------------------------------------------ #
    # Retrieval-contract views (ips_r.core.types)
    # ------------------------------------------------------------------ #
    def queries(self) -> list[core.Query]:
        """Return the questions as :class:`~ips_r.core.types.Query` objects."""
        return [question.to_query() for question in self.load_questions()]

    def documents(self) -> list[core.Document]:
        """Return the documents as :class:`~ips_r.core.types.Document` objects."""
        return [document.to_core() for document in self.load_documents()]

    def references(self) -> list[core.Reference]:
        """Return gold references aligned with :meth:`queries` by ``query_id``."""
        return [question.to_reference() for question in self.load_questions()]

    # ------------------------------------------------------------------ #
    # Implementation helpers
    # ------------------------------------------------------------------ #
    def _report(
        self,
        code: IssueCode,
        message: str,
        *,
        severity: Severity = Severity.ERROR,
        record_id: str | None = None,
        location: str | None = None,
    ) -> None:
        """Record an ingestion issue (and raise in strict mode for errors).

        Raises:
            ComponentError: If ``strict`` is enabled and ``severity`` is ``ERROR``.
        """
        self._record(ValidationIssue(code, severity, message, record_id, location))

    def _record(self, issue: ValidationIssue) -> None:
        """Record a prepared issue; usable as an :data:`ips_r.data.io.IssueSink`.

        Raises:
            ComponentError: If ``strict`` is enabled and the issue is an error.
        """
        if self._issues.add(issue):
            self._logger.debug("dataset.issue", **issue.to_dict())
        if self._strict and issue.severity is Severity.ERROR:
            where = f" at {issue.location}" if issue.location else ""
            raise ComponentError(self.name, f"strict mode: {issue.message}{where}")

    def _record_all(self, issues: Iterable[ValidationIssue]) -> bool:
        """Record several issues; return ``True`` if none of them is an error."""
        ok = True
        for issue in issues:
            self._record(issue)
            ok = ok and issue.severity is not Severity.ERROR
        return ok

    def _build[T](
        self, factory: Callable[[], T], *, record_id: str | None, location: str | None
    ) -> T | None:
        """Construct a model, reporting a ``malformed_record`` instead of raising ``ValueError``.

        Raises:
            ComponentError: In strict mode, if construction fails.
        """
        try:
            return factory()
        except ValueError as exc:
            self._report(
                IssueCode.MALFORMED_RECORD, str(exc), record_id=record_id, location=location
            )
            return None

    def _path_param(self, key: str, *, directory: bool = False) -> Path:
        """Return parameter ``key`` as an existing file (or directory) path.

        Raises:
            ConfigurationError: If the parameter is missing or the path does not exist.
        """
        path = Path(self.str_param(key)).expanduser()
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            kind = "directory" if directory else "file"
            raise ConfigurationError(
                f"dataset '{self.name}' parameter '{key}': {kind} not found: {path}"
            )
        return path

    def _sources(self) -> tuple[Path, ...]:
        """Source files or directories, for provenance. Override in implementations."""
        return ()

    def _teardown(self) -> None:
        self._documents_cache = None
        self._questions_cache = None
        self._release()

    def _release(self) -> None:  # noqa: B027 - intentionally optional hook
        """Hook to free cached raw data on teardown. Default is a no-op."""

    @abstractmethod
    def _iter_documents(self) -> Iterable[Document]:
        """Yield canonical documents converted from the source."""
        raise NotImplementedError

    @abstractmethod
    def _iter_questions(self) -> Iterable[Question]:
        """Yield canonical questions converted from the source."""
        raise NotImplementedError


DATASETS: Registry[AbstractDataset] = Registry("dataset", AbstractDataset)
"""Registry of dataset implementations."""
