"""MS MARCO passage-ranking adapter (TSV files, optionally gzipped).

Source files:

- ``collection_path``: ``pid<TAB>passage`` (one document per passage, id = pid)
- ``queries_path`` (optional): ``qid<TAB>query``
- ``qrels_path`` (optional): TREC ``qid<TAB>0<TAB>pid<TAB>relevance`` or ``qid<TAB>pid``

Questions carry no answer strings; relevant passages (relevance > 0) become
``supporting_document_ids``. With ``only_judged_queries`` (and a qrels file), queries
without any relevant passage are skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from ips_r.data.adapters.common import unique
from ips_r.data.base import DATASETS, AbstractDataset
from ips_r.data.io import iter_tsv
from ips_r.data.issues import IssueCode
from ips_r.data.models import Document, DocumentMetadata, Question
from ips_r.data.validation import SchemaTracker


@DATASETS.register("ms_marco")
class MSMarcoDataset(AbstractDataset):
    """MS MARCO passage ranking.

    Parameters:
        collection_path: Passage collection TSV.
        queries_path: Queries TSV (``null`` = no questions).
        qrels_path: Relevance judgements (``null`` = no supporting ids).
        only_judged_queries: Skip queries without relevant passages (requires qrels).
    """

    description: ClassVar[str] = "MS MARCO passage ranking; one document per passage."
    homepage: ClassVar[str | None] = "https://microsoft.github.io/msmarco/"
    license: ClassVar[str | None] = "MS MARCO terms of use (non-commercial research)"
    language: ClassVar[str | None] = "en"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._collection = Path(self.str_param("collection_path"))
        queries = self.optional_str_param("queries_path")
        qrels = self.optional_str_param("qrels_path")
        self._queries = Path(queries) if queries else None
        self._qrels = Path(qrels) if qrels else None
        self._only_judged = self.bool_param("only_judged_queries", True)

    def _setup(self) -> None:
        self._collection = self._path_param("collection_path")
        if self._queries is not None:
            self._queries = self._path_param("queries_path")
        if self._qrels is not None:
            self._qrels = self._path_param("qrels_path")

    def _sources(self) -> tuple[Path, ...]:
        return tuple(p for p in (self._collection, self._queries, self._qrels) if p is not None)

    def _rows(
        self, path: Path, name: str, columns: tuple[int, ...]
    ) -> Iterator[tuple[str, list[str]]]:
        """Yield rows with an accepted column count and non-empty first column."""
        tracker = SchemaTracker(name)
        for location, row in iter_tsv(path, sink=self._record):
            self._record_all(tracker.observe_columns(len(row), location=location))
            if len(row) not in columns:
                self._report(
                    IssueCode.MALFORMED_RECORD,
                    f"{name}: expected {' or '.join(map(str, columns))} columns, got {len(row)}",
                    location=location,
                )
            elif not row[0].strip():
                self._report(IssueCode.MISSING_FIELD, f"{name}: empty id column", location=location)
            else:
                yield location, row

    def _iter_documents(self) -> Iterator[Document]:
        for location, (pid, passage) in self._rows(self._collection, "ms_marco.collection", (2,)):
            yield Document(
                id=pid.strip(),
                text=passage,
                metadata=DocumentMetadata(language=self.language),
                source=location,
            )

    def _relevant(self) -> dict[str, list[str]]:
        """Map qid → relevant pids (relevance > 0) from the qrels file."""
        relevant: dict[str, list[str]] = {}
        if self._qrels is None:
            return relevant
        for location, row in self._rows(self._qrels, "ms_marco.qrels", (2, 4)):
            qid, pid = row[0].strip(), row[1 if len(row) == 2 else 2].strip()
            grade = "1" if len(row) == 2 else row[3].strip()
            try:
                if int(grade) <= 0:
                    continue
            except ValueError:
                self._report(
                    IssueCode.MALFORMED_RECORD,
                    f"ms_marco.qrels: relevance {grade!r} is not an integer",
                    location=location,
                )
                continue
            relevant.setdefault(qid, []).append(pid)
        return relevant

    def _iter_questions(self) -> Iterator[Question]:
        if self._queries is None:
            return
        relevant = self._relevant()
        skip_unjudged = self._only_judged and self._qrels is not None
        for _, (qid, text) in self._rows(self._queries, "ms_marco.queries", (2,)):
            qid = qid.strip()
            if skip_unjudged and qid not in relevant:
                continue
            yield Question(
                id=qid,
                text=text,
                supporting_document_ids=unique(relevant.get(qid, ())),
                question_type="ranking",
            )
