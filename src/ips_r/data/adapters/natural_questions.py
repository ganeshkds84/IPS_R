"""Natural Questions adapter (JSON lines, optionally gzipped).

Both official layouts are accepted and may not be mixed within one file (mixing is
reported as ``inconsistent_schema``):

- *simplified* (``v1.0-simplified``): ``document_text`` is a space-separated token string
  that includes HTML tags such as ``<P>``;
- *original*: ``document_tokens`` is a list of ``{"token", "html_token"}`` objects.

Each example yields one document (the Wikipedia page, id = ``example_id``) and one
question with the same id. Short answers (token spans) and yes/no answers become
answer strings; HTML tokens are removed when ``strip_html`` is true.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

from ips_r.data.adapters.common import ID, STR, unique
from ips_r.data.base import DATASETS, AbstractDataset
from ips_r.data.io import iter_jsonl
from ips_r.data.issues import IssueCode, Severity
from ips_r.data.models import Document, DocumentMetadata, Question
from ips_r.data.validation import RecordSchema, SchemaTracker

ARRAY: Final = frozenset({"array"})
EXAMPLE = RecordSchema(
    "natural_questions.example",
    {"example_id": ID, "question_text": STR, "annotations": ARRAY},
    {"document_text": STR, "document_tokens": ARRAY, "document_url": STR, "document_title": STR},
)
_HTML_TOKEN: Final = re.compile(r"^<[^>]*>$")
_YES_NO: Final = {"YES": "yes", "NO": "no"}

Token = tuple[str, bool]
"""``(token, is_html)``."""


@DATASETS.register("natural_questions")
class NaturalQuestionsDataset(AbstractDataset):
    """Google Natural Questions.

    Parameters:
        path: JSON-lines file (``.jsonl`` or ``.jsonl.gz``).
        strip_html: Remove HTML tokens from document text and answers.
    """

    description: ClassVar[str] = "Natural Questions; one Wikipedia page per example."
    homepage: ClassVar[str | None] = "https://ai.google.com/research/NaturalQuestions"
    license: ClassVar[str | None] = "CC BY-SA 3.0"
    language: ClassVar[str | None] = "en"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._path = Path(self.str_param("path"))
        self._strip_html = self.bool_param("strip_html", True)

    def _setup(self) -> None:
        self._path = self._path_param("path")

    def _sources(self) -> tuple[Path, ...]:
        return (self._path,)

    def _examples(self) -> Iterator[tuple[str, str, Mapping[str, Any], list[Token]]]:
        """Yield ``(location, id, example, tokens)`` for valid examples."""
        tracker = SchemaTracker(
            "natural_questions.example",
            ignore=("document_url", "document_title", "long_answer_candidates"),
        )
        for location, record in iter_jsonl(self._path, sink=self._record):
            raw_id = record.get("example_id") if isinstance(record, Mapping) else None
            record_id = str(raw_id) if isinstance(raw_id, str | int) else None
            if not self._record_all(EXAMPLE.check(record, record_id=record_id, location=location)):
                continue
            assert record_id is not None  # guaranteed by the schema check
            self._record_all(tracker.observe(record, location=location))
            tokens = self._tokens(record, record_id, location)
            if tokens is not None:
                yield location, record_id, record, tokens

    def _tokens(
        self, record: Mapping[str, Any], record_id: str, location: str
    ) -> list[Token] | None:
        """Extract document tokens from either layout."""
        if isinstance(record.get("document_text"), str):
            return [
                (tok, bool(_HTML_TOKEN.match(tok))) for tok in record["document_text"].split(" ")
            ]
        tokens = record.get("document_tokens")
        if isinstance(tokens, list):
            if all(isinstance(t, Mapping) and isinstance(t.get("token"), str) for t in tokens):
                return [(t["token"], bool(t.get("html_token", False))) for t in tokens]
            self._report(
                IssueCode.MALFORMED_RECORD,
                "document_tokens entries need a 'token' string",
                record_id=record_id,
                location=location,
            )
            return None
        self._report(
            IssueCode.MISSING_FIELD,
            "natural_questions.example: needs 'document_text' or 'document_tokens'",
            record_id=record_id,
            location=location,
        )
        return None

    def _join(self, tokens: Sequence[Token]) -> str:
        """Join tokens into text, dropping HTML tokens when configured."""
        return " ".join(tok for tok, html in tokens if tok and not (html and self._strip_html))

    def _iter_documents(self) -> Iterator[Document]:
        for location, record_id, record, tokens in self._examples():
            document = self._build(
                lambda: Document(
                    id=record_id,
                    text=self._join(tokens),
                    metadata=DocumentMetadata(
                        title=record.get("document_title"),
                        url=record.get("document_url"),
                        language=self.language,
                    ),
                    source=str(self._path),
                ),
                record_id=record_id,
                location=location,
            )
            if document is not None:
                yield document

    def _iter_questions(self) -> Iterator[Question]:
        for location, record_id, record, tokens in self._examples():
            answers, has_long = self._answers(record["annotations"], tokens, record_id, location)
            yes_no = any(a in _YES_NO.values() for a in answers)
            question_type = (
                "yes_no" if yes_no else "short" if answers else "long" if has_long else "none"
            )
            question = self._build(
                lambda: Question(
                    id=record_id,
                    text=record["question_text"],
                    answers=unique(answers),
                    supporting_document_ids=(record_id,),
                    question_type=question_type,
                    metadata={"has_long_answer": has_long},
                ),
                record_id=record_id,
                location=location,
            )
            if question is not None:
                yield question

    def _answers(
        self, annotations: Sequence[Any], tokens: Sequence[Token], record_id: str, location: str
    ) -> tuple[list[str], bool]:
        """Return answer strings and whether any long answer exists."""
        answers: list[str] = []
        has_long = False
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                self._report(
                    IssueCode.MALFORMED_RECORD,
                    "annotation must be an object",
                    severity=Severity.WARNING,
                    record_id=record_id,
                    location=location,
                )
                continue
            long_answer = annotation.get("long_answer")
            if isinstance(long_answer, Mapping) and int(long_answer.get("start_token", -1)) >= 0:
                has_long = True
            if (yes_no := _YES_NO.get(str(annotation.get("yes_no_answer", "NONE")))) is not None:
                answers.append(yes_no)
            for short in annotation.get("short_answers") or []:
                start = short.get("start_token") if isinstance(short, Mapping) else None
                end = short.get("end_token") if isinstance(short, Mapping) else None
                if not (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(tokens)
                ):
                    self._report(
                        IssueCode.MALFORMED_RECORD,
                        f"short answer span [{start}, {end}) outside document of {len(tokens)} tokens",
                        severity=Severity.WARNING,
                        record_id=record_id,
                        location=location,
                    )
                    continue
                text = self._join(tokens[start:end])
                if text:
                    answers.append(text)
        return answers, has_long
