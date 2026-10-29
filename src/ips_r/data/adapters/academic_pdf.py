"""Academic PDF collection adapter: a directory of papers (+ optional questions).

Every PDF matched by ``glob`` under ``directory`` is parsed by
:class:`~ips_r.data.pdf.PDFParser` into a structured document (title, authors,
headings, sections, paragraphs, captions, tables, references, page numbers) with
extracted bibliographic metadata. Unparseable files are reported as
``corrupted_pdf`` and skipped. Files are processed in sorted order.

Optional questions (``questions_path``, JSON lines)::

    {"id": "q1", "question": "...", "answers": ["..."], "document_ids": ["paper-a"]}
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Final

from ips_r.core.exceptions import ConfigurationError, PDFParseError
from ips_r.data.adapters.common import ID, STR, unique
from ips_r.data.base import DATASETS, AbstractDataset
from ips_r.data.io import iter_jsonl
from ips_r.data.issues import IssueCode
from ips_r.data.models import Document, Question
from ips_r.data.pdf import PDFParser
from ips_r.data.validation import RecordSchema, SchemaTracker

ARRAY: Final = frozenset({"array"})
QUESTION = RecordSchema(
    "academic_pdf.question",
    {"id": ID, "question": STR},
    {"answers": ARRAY, "document_ids": ARRAY, "type": STR},
)
ID_STRATEGIES: Final = ("stem", "relative_path")


@DATASETS.register("academic_pdf")
class AcademicPDFDataset(AbstractDataset):
    """A local collection of academic papers in PDF format.

    Parameters:
        directory: Root directory of the collection.
        glob: File pattern relative to ``directory`` (default ``**/*.pdf``).
        id_strategy: ``stem`` (file name without suffix) or ``relative_path``.
        questions_path: Optional JSON-lines question file.
        parser: :class:`~ips_r.data.pdf.PDFParser` configuration.
    """

    description: ClassVar[str] = "Academic PDF collection parsed into structured documents."

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._directory = Path(self.str_param("directory"))
        self._glob = self.str_param("glob", "**/*.pdf")
        self._id_strategy = self.str_param("id_strategy", "stem")
        if self._id_strategy not in ID_STRATEGIES:
            raise ConfigurationError(
                f"dataset '{self.name}' id_strategy must be one of {ID_STRATEGIES}, "
                f"got {self._id_strategy!r}"
            )
        questions = self.optional_str_param("questions_path")
        self._questions_path = Path(questions) if questions else None
        self._parser = PDFParser.from_config(self.param("parser", None))

    def _setup(self) -> None:
        self._directory = self._path_param("directory", directory=True)
        if self._questions_path is not None:
            self._questions_path = self._path_param("questions_path")

    def _sources(self) -> tuple[Path, ...]:
        return tuple(p for p in (self._directory, self._questions_path) if p is not None)

    def files(self) -> list[Path]:
        """Return the PDF files of the collection in deterministic order."""
        if not self.is_ready:
            self.setup()
        files = (p for p in self._directory.glob(self._glob) if p.is_file())
        return sorted(files, key=lambda p: p.relative_to(self._directory).as_posix())

    def document_id(self, path: Path) -> str:
        """Return the document id for ``path`` according to ``id_strategy``."""
        if self._id_strategy == "stem":
            return path.stem
        return path.relative_to(self._directory).with_suffix("").as_posix()

    def _iter_documents(self) -> Iterator[Document]:
        for path in self.files():
            doc_id = self.document_id(path)
            try:
                yield self._parser.parse(path, doc_id)
            except PDFParseError as exc:
                self._report(
                    IssueCode.CORRUPTED_PDF, str(exc), record_id=doc_id, location=str(path)
                )

    def _iter_questions(self) -> Iterator[Question]:
        if self._questions_path is None:
            return
        tracker = SchemaTracker("academic_pdf.question", ignore=("answers", "document_ids", "type"))
        for location, record in iter_jsonl(self._questions_path, sink=self._record):
            raw_id = record.get("id") if isinstance(record, Mapping) else None
            record_id = str(raw_id) if isinstance(raw_id, str | int) else None
            if not self._record_all(QUESTION.check(record, record_id=record_id, location=location)):
                continue
            self._record_all(tracker.observe(record, location=location))
            answers = record.get("answers") or []
            doc_ids = record.get("document_ids") or []
            if not all(isinstance(a, str) for a in [*answers, *doc_ids]):
                self._report(
                    IssueCode.MALFORMED_RECORD,
                    "answers and document_ids must be strings",
                    record_id=record_id,
                    location=location,
                )
                continue
            question = self._build(
                lambda: Question(
                    id=str(record["id"]),
                    text=record["question"],
                    answers=unique(answers),
                    supporting_document_ids=unique(doc_ids),
                    question_type=record.get("type"),
                ),
                record_id=record_id,
                location=location,
            )
            if question is not None:
                yield question
