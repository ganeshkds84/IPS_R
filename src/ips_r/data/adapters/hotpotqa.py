"""HotpotQA adapter (distractor / fullwiki JSON): multi-hop questions over Wikipedia paragraphs.

Source format (``params.path``)::

    [{"_id": "...", "question": "...", "answer": "...", "type": "bridge", "level": "hard",
      "supporting_facts": [["Title", 0], ...],
      "context": [["Title", ["Sentence 0.", " Sentence 1."]], ...]}]

Each distinct context paragraph becomes one document with id = Wikipedia title.
Paragraphs repeated across questions are emitted once; a title that reappears with
*different* text is reported as ``duplicate_id`` (the first version is kept).
Sentence offsets are kept in ``Document.extra["sentence_spans"]``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Final

from ips_r.data.adapters.common import STR, unique
from ips_r.data.base import DATASETS, AbstractDataset
from ips_r.data.io import load_json
from ips_r.data.issues import IssueCode, Severity
from ips_r.data.models import Document, DocumentMetadata, Question
from ips_r.data.text import stable_hash
from ips_r.data.validation import RecordSchema, SchemaTracker

ARRAY: Final = frozenset({"array"})
EXAMPLE = RecordSchema(
    "hotpotqa.example",
    {"_id": STR, "question": STR, "context": ARRAY},
    {"answer": STR, "supporting_facts": ARRAY, "type": STR, "level": STR},
)


def _paragraph(entry: Any) -> tuple[str, list[str]] | None:
    """Validate a ``[title, [sentences...]]`` context entry."""
    if (
        isinstance(entry, list | tuple)
        and len(entry) == 2
        and isinstance(entry[0], str)
        and entry[0]
        and isinstance(entry[1], list)
        and all(isinstance(s, str) for s in entry[1])
    ):
        return entry[0], list(entry[1])
    return None


@DATASETS.register("hotpotqa")
class HotpotQADataset(AbstractDataset):
    """HotpotQA multi-hop question answering.

    Parameters:
        path: HotpotQA JSON file (a list of examples).
    """

    description: ClassVar[str] = "HotpotQA multi-hop QA; one document per Wikipedia paragraph."
    homepage: ClassVar[str | None] = "https://hotpotqa.github.io/"
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

    def _examples(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        """Yield ``(location, example)`` for schema-valid examples."""
        if self._raw is None:
            self._raw = load_json(self._path, sink=self._record)
            if self._raw is None:
                self._raw = []
        if not isinstance(self._raw, list):
            self._report(
                IssueCode.MALFORMED_RECORD,
                "top level must be an array of examples",
                location=str(self._path),
            )
            return
        tracker = SchemaTracker(
            "hotpotqa.example", ignore=("answer", "supporting_facts", "type", "level")
        )
        for index, example in enumerate(self._raw):
            location = f"{self._path}[{index}]"
            qid = example.get("_id") if isinstance(example, Mapping) else None
            record_id = qid if isinstance(qid, str) else None
            if self._record_all(EXAMPLE.check(example, record_id=record_id, location=location)):
                self._record_all(tracker.observe(example, location=location))
                yield location, example

    def _iter_documents(self) -> Iterator[Document]:
        seen: dict[str, str] = {}
        for location, example in self._examples():
            for c_index, entry in enumerate(example["context"]):
                parsed = _paragraph(entry)
                if parsed is None:
                    self._report(
                        IssueCode.MALFORMED_RECORD,
                        "context entry must be [title, [sentences]]",
                        record_id=example["_id"],
                        location=f"{location}.context[{c_index}]",
                    )
                    continue
                title, sentences = parsed
                text = "".join(sentences)
                digest = stable_hash([text])
                if title in seen:
                    if seen[title] != digest:
                        self._report(
                            IssueCode.DUPLICATE_ID,
                            "title reappears with different text; keeping the first version",
                            severity=Severity.WARNING,
                            record_id=title,
                            location=f"{location}.context[{c_index}]",
                        )
                    continue
                seen[title] = digest
                yield Document(
                    id=title,
                    text=text,
                    metadata=DocumentMetadata(title=title, language=self.language),
                    source=str(self._path),
                    extra={"sentence_spans": _sentence_spans(sentences)},
                )

    def _iter_questions(self) -> Iterator[Question]:
        for location, example in self._examples():
            facts = self._supporting_facts(example, location)
            answer = example.get("answer")
            question = self._build(
                lambda: Question(
                    id=example["_id"],
                    text=example["question"],
                    answers=(answer,) if isinstance(answer, str) and answer else (),
                    supporting_document_ids=unique(title for title, _ in facts),
                    question_type=example.get("type"),
                    metadata={
                        "level": example.get("level"),
                        "supporting_facts": [list(f) for f in facts],
                    },
                ),
                record_id=example["_id"],
                location=location,
            )
            if question is not None:
                yield question

    def _supporting_facts(self, example: Mapping[str, Any], location: str) -> list[tuple[str, int]]:
        """Return valid ``(title, sentence index)`` pairs, reporting malformed ones."""
        facts: list[tuple[str, int]] = []
        for f_index, fact in enumerate(example.get("supporting_facts") or []):
            if (
                isinstance(fact, list | tuple)
                and len(fact) == 2
                and isinstance(fact[0], str)
                and isinstance(fact[1], int)
                and not isinstance(fact[1], bool)
            ):
                facts.append((fact[0], fact[1]))
            else:
                self._report(
                    IssueCode.MALFORMED_RECORD,
                    "supporting fact must be [title, sentence_index]",
                    severity=Severity.WARNING,
                    record_id=example["_id"],
                    location=f"{location}.supporting_facts[{f_index}]",
                )
        return facts


def _sentence_spans(sentences: list[str]) -> list[list[int]]:
    """Character spans of each sentence within the concatenated paragraph."""
    spans: list[list[int]] = []
    offset = 0
    for sentence in sentences:
        spans.append([offset, offset + len(sentence)])
        offset += len(sentence)
    return spans
