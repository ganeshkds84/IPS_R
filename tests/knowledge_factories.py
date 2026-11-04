"""Synthetic structured documents and an in-memory dataset for knowledge-pipeline tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from ips_r.data import AbstractDataset
from ips_r.data.models import Document, DocumentMetadata, ElementDraft, ElementType, Question

BODY = (
    "Retrieval systems rewrite queries to improve recall [1]. Prior work (Smith et al., 2020) "
    "studied this problem in depth. See Figure 1 and Table 1 for details, and Section 2.1. "
)


def paper(doc_id: str = "paper", *, sections: int = 2, body_repeats: int = 2) -> Document:
    """A paper-like document using every element type, with nested and appendix sections."""
    e = ElementType
    drafts: list[ElementDraft] = [
        ElementDraft(e.TITLE, "A Synthetic Study of Intent", page=1, bbox=(70, 60, 400, 80)),
        ElementDraft(e.AUTHORS, "Grace Hopper, Alan Turing", page=1),
        ElementDraft(e.HEADING, "Abstract", page=1, level=1),
        ElementDraft(e.ABSTRACT, "We study the eﬃcient “preservation” of intent.", page=1),
    ]
    for number in range(1, sections + 1):
        drafts += [
            ElementDraft(e.HEADING, f"{number} Section {number}", page=1, level=1),
            ElementDraft(e.PARAGRAPH, BODY * body_repeats, page=1, bbox=(70, 100, 500, 200)),
            ElementDraft(e.HEADING, f"{number}.1 Details {number}", page=1, level=2),
            ElementDraft(e.PARAGRAPH, f"Details of part {number} are given here. " * 3, page=1),
            ElementDraft(e.EQUATION, f"y = a x + b ({number})", page=1),
        ]
    drafts += [
        ElementDraft(e.CAPTION, "Figure 1: The pipeline overview.", page=2),
        ElementDraft(e.CAPTION, "Table 1: Main results.", page=2),
        ElementDraft(
            e.TABLE,
            "Model | Score\nBM25 | 0.41",
            page=2,
            rows=(("Model", "Score"), ("BM25", "0.41")),
        ),
        ElementDraft(e.FOOTNOTE, "1 Code is available.", page=2),
        ElementDraft(e.HEADING, "References", page=2, level=1),
        ElementDraft(e.REFERENCE, "[1] J. Doe. Query rewriting. 2019.", page=2),
        ElementDraft(
            e.REFERENCE,
            "[2] Smith, A. and Lee, B. Intent models. In Proc. ACL, 2020. doi:10.1000/xyz.1",
            page=2,
        ),
        ElementDraft(e.HEADING, "Appendix A Extra Results", page=2, level=1),
        ElementDraft(e.HEADING, "A.1 More Tables", page=2, level=2),
        ElementDraft(
            e.PARAGRAPH, "Additional results are reported in Eq. (1) and Appendix A.", page=2
        ),
    ]
    return Document.from_elements(
        doc_id,
        drafts,
        metadata=DocumentMetadata(
            title="A Synthetic Study of Intent", authors=("Grace Hopper",), year=2024
        ),
        source=f"/data/{doc_id}.pdf",
        num_pages=2,
    )


def big_table_document(doc_id: str = "tables", rows: int = 200) -> Document:
    """A document with one large table that must be split row-wise."""
    table_rows = [("id", "value")] + [(str(i), f"value-{i}") for i in range(rows)]
    text = "\n".join(" | ".join(row) for row in table_rows)
    return Document.from_elements(
        doc_id,
        [
            ElementDraft(ElementType.HEADING, "1 Data", page=1, level=1),
            ElementDraft(ElementType.CAPTION, "Table 1: All values.", page=1),
            ElementDraft(ElementType.TABLE, text, page=1, rows=tuple(table_rows)),
        ],
        num_pages=1,
    )


class InMemoryDataset(AbstractDataset):
    """Dataset backed by lists (tests only)."""

    def __init__(
        self,
        documents: Sequence[Document],
        questions: Sequence[Question] = (),
        config: Any = None,
        *,
        name: str | None = "memory",
    ) -> None:
        super().__init__(config, name=name)
        self._documents = list(documents)
        self._questions = list(questions)

    def _iter_documents(self) -> Iterator[Document]:
        yield from self._documents

    def _iter_questions(self) -> Iterator[Question]:
        yield from self._questions
