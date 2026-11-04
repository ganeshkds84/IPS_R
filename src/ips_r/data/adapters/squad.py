"""SQuAD v1.1 / v2.0 adapter: one document per paragraph, extractive questions.

Source format (``params.path``)::

    {"version": "v2.0", "data": [{"title": "...", "paragraphs": [
        {"context": "...", "qas": [{"id": "...", "question": "...",
          "answers": [{"text": "...", "answer_start": 0}], "is_impossible": false}]}]}]}

Document id: ``"<title>#<paragraph index>"``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Final

from ips_r.data.adapters.common import STR, to_title, unique
from ips_r.data.base import DATASETS, AbstractDataset
from ips_r.data.io import load_json
from ips_r.data.issues import IssueCode
from ips_r.data.models import Document, DocumentMetadata, Question
from ips_r.data.validation import RecordSchema, SchemaTracker

ARRAY: Final = frozenset({"array"})
BOOL: Final = frozenset({"bool"})
NUMBER: Final = frozenset({"number"})

ARTICLE = RecordSchema("squad.article", {"paragraphs": ARRAY}, {"title": STR})
PARAGRAPH = RecordSchema("squad.paragraph", {"context": STR, "qas": ARRAY})
QUESTION = RecordSchema(
    "squad.question",
    {"id": STR, "question": STR, "answers": ARRAY},
    {"is_impossible": BOOL, "plausible_answers": ARRAY},
)
ANSWER = RecordSchema("squad.answer", {"text": STR}, {"answer_start": NUMBER})

_Paragraph = tuple[str, str, str, Mapping[str, Any]]  # location, doc id, title, paragraph


@DATASETS.register("squad")
class SquadDataset(AbstractDataset):
    """Stanford Question Answering Dataset (v1.1 and v2.0 JSON).

    Parameters:
        path: SQuAD JSON file.
    """

    description: ClassVar[str] = "SQuAD reading-comprehension dataset; one document per paragraph."
    homepage: ClassVar[str | None] = "https://rajpurkar.github.io/SQuAD-explorer/"
    license: ClassVar[str | None] = "CC BY-SA 4.0"
    language: ClassVar[str | None] = "en"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._path = Path(self.str_param("path"))
        self._raw: Any | None = None

    def _setup(self) -> None:
        self._path = self._path_param("path")

    def _sources(self) -> tuple[Path, ...]:
        return (self._path,)

    def _release(self) -> None:
        self._raw = None

    def _articles(self) -> list[Any]:
        """Return the (cached) list of article records."""
        if self._raw is None:
            self._raw = load_json(self._path, sink=self._record) or {}
        data = self._raw.get("data") if isinstance(self._raw, Mapping) else None
        if not isinstance(data, list):
            self._report(
                IssueCode.MALFORMED_RECORD,
                "top level must be an object with a 'data' array",
                location=str(self._path),
            )
            return []
        return data

    def _paragraphs(self) -> Iterator[_Paragraph]:
        """Yield valid paragraphs with their document id and title."""
        for a_index, article in enumerate(self._articles()):
            location = f"{self._path}#data[{a_index}]"
            if not self._record_all(ARTICLE.check(article, location=location)):
                continue
            title = article.get("title") or f"article-{a_index}"
            for p_index, paragraph in enumerate(article["paragraphs"]):
                p_location = f"{location}.paragraphs[{p_index}]"
                doc_id = f"{title}#{p_index}"
                if self._record_all(
                    PARAGRAPH.check(paragraph, record_id=doc_id, location=p_location)
                ):
                    yield p_location, doc_id, title, paragraph

    def _document(self, doc_id: str, title: str, paragraph: Mapping[str, Any]) -> Document:
        """Build the document of a paragraph (raises ``ValueError`` on invalid fields)."""
        return Document(
            id=doc_id,
            text=paragraph["context"],
            metadata=DocumentMetadata(title=to_title(title), language=self.language),
            source=str(self._path),
            extra={"article_title": title},
        )

    def _iter_documents(self) -> Iterator[Document]:
        for location, doc_id, title, paragraph in self._paragraphs():
            document = self._build(
                partial(self._document, doc_id, title, paragraph),
                record_id=doc_id,
                location=location,
            )
            if document is not None:
                yield document

    def _iter_questions(self) -> Iterator[Question]:
        tracker = SchemaTracker("squad.question", ignore=("plausible_answers",))
        for location, doc_id, _, paragraph in self._paragraphs():
            for q_index, qa in enumerate(paragraph["qas"]):
                q_location = f"{location}.qas[{q_index}]"
                qid = qa.get("id") if isinstance(qa, Mapping) else None
                record_id = qid if isinstance(qid, str) else None
                if not self._record_all(
                    QUESTION.check(qa, record_id=record_id, location=q_location)
                ):
                    continue
                self._record_all(tracker.observe(qa, location=q_location))
                question = self._question(qa, doc_id, q_location)
                if question is not None:
                    yield question

    def _question(self, qa: Mapping[str, Any], doc_id: str, location: str) -> Question | None:
        """Convert one validated question record."""
        answers: list[str] = []
        starts: list[int] = []
        for answer in qa["answers"]:
            if self._record_all(ANSWER.check(answer, record_id=qa["id"], location=location)):
                answers.append(answer["text"])
                if isinstance(answer.get("answer_start"), int):
                    starts.append(answer["answer_start"])
        impossible = bool(qa.get("is_impossible", False))
        return self._build(
            lambda: Question(
                id=qa["id"],
                text=qa["question"],
                answers=unique(answers),
                supporting_document_ids=(doc_id,),
                question_type="unanswerable" if impossible else "extractive",
                metadata={"answer_starts": starts, "is_impossible": impossible},
            ),
            record_id=qa["id"],
            location=location,
        )
