"""Tests for the canonical dataset models."""

from __future__ import annotations

from typing import Any

import pytest

from ips_r.data.models import (
    ELEMENT_SEPARATOR,
    Chunk,
    DatasetMetadata,
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementDraft,
    ElementType,
    Question,
)


def _doc() -> Document:
    return Document.from_elements(
        "paper",
        [
            ElementDraft(ElementType.TITLE, "A Title", page=1),
            ElementDraft(ElementType.HEADING, "1 Intro", page=1, section="1 Intro", level=1),
            ElementDraft(ElementType.PARAGRAPH, "Body text.", page=2, section="1 Intro"),
            ElementDraft(ElementType.TABLE, "a | b", page=2, rows=(("a", "b"),)),
        ],
        metadata=DocumentMetadata(title="A Title", authors=("Ada",), year=2024, keywords=("x",)),
        source="paper.pdf",
        num_pages=2,
        extra={"k": 1},
    )


def test_from_elements_assembles_text_and_offsets() -> None:
    doc = _doc()
    assert doc.text == ELEMENT_SEPARATOR.join(["A Title", "1 Intro", "Body text.", "a | b"])
    for element in doc.elements:
        assert doc.text[element.start : element.end] == element.text
    assert doc.title == "A Title"
    assert doc.metadata.authors == ("Ada",)
    assert doc.elements[3].rows == (("a", "b"),)
    assert [e.type for e in doc.elements_in(0, 12)] == [ElementType.TITLE, ElementType.HEADING]


def test_document_round_trip_and_core_view() -> None:
    doc = _doc()
    assert Document.from_dict(doc.to_dict()) == doc
    core = doc.to_core()
    assert core.id == "paper"
    assert core.metadata["title"] == "A Title"
    assert core.metadata["k"] == 1
    with pytest.raises(TypeError):
        doc.extra["k"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": ""},
        {"id": 5},
        {"text": None},
        {"num_pages": 0},
        {"metadata": {"title": "x"}},
        {"source": 3},
        {"extra": [1]},
    ],
)
def test_document_rejects_invalid_fields(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {"id": "d", "text": "t"}
    with pytest.raises(ValueError):
        Document(**{**base, **kwargs})


def test_document_rejects_inconsistent_elements() -> None:
    element = DocumentElement(ElementType.PARAGRAPH, "abc", 0, 3, page=2)
    with pytest.raises(ValueError, match="does not match"):
        Document(id="d", text="xyz", elements=(element,))
    with pytest.raises(ValueError, match="sorted"):
        Document(id="d", text="ab", elements=(element,))
    with pytest.raises(ValueError, match="exceeds"):
        Document(id="d", text="abc", elements=(element,), num_pages=1)
    overlapping = (element, DocumentElement(ElementType.PARAGRAPH, "c", 2, 3))
    with pytest.raises(ValueError, match="sorted"):
        Document(id="d", text="abc", elements=overlapping)


def test_element_validation() -> None:
    with pytest.raises(ValueError, match="span"):
        DocumentElement(ElementType.PARAGRAPH, "abc", 0, 5)
    with pytest.raises(ValueError):
        ElementDraft(ElementType.PARAGRAPH, "   ")
    with pytest.raises(ValueError):
        ElementDraft("nonsense", "text")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ElementDraft(ElementType.HEADING, "h", level=0)
    with pytest.raises(ValueError):
        ElementDraft(ElementType.TABLE, "t", rows=(("a", 1),))  # type: ignore[arg-type]
    assert ElementDraft("caption", "Fig. 1").type is ElementType.CAPTION  # type: ignore[arg-type]


def test_metadata_validation_and_round_trip() -> None:
    meta = DocumentMetadata(title="T", authors=("A", "B"), year=2020, doi="10.1/x", url="https://x")
    assert DocumentMetadata.from_dict(meta.to_dict()) == meta
    with pytest.raises(ValueError):
        DocumentMetadata(year=True)
    with pytest.raises(ValueError):
        DocumentMetadata(authors="Ada")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DocumentMetadata(title=1)  # type: ignore[arg-type]


def test_chunk_round_trip_and_core_view() -> None:
    chunk = Chunk(
        id="d::00000",
        document_id="d",
        text="abc",
        index=0,
        start=0,
        end=3,
        strategy="token",
        page_start=1,
        page_end=2,
        section="S",
        metadata={"m": 1},
    )
    assert Chunk.from_dict(chunk.to_dict()) == chunk
    assert chunk.num_chars == 3
    core = chunk.to_core()
    assert core.id == "d::00000"
    assert core.metadata["document_id"] == "d"
    assert core.metadata["m"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": ""},
        {"end": 9},
        {"index": -1},
        {"page_start": 1},
        {"page_start": 3, "page_end": 2},
        {"strategy": ""},
    ],
)
def test_chunk_rejects_invalid_fields(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "id": "c",
        "document_id": "d",
        "text": "abc",
        "index": 0,
        "start": 0,
        "end": 3,
        "strategy": "s",
    }
    with pytest.raises(ValueError):
        Chunk(**{**base, **kwargs})


def test_question_round_trip_and_core_views() -> None:
    question = Question(
        id="q",
        text="Why?",
        answers=("because",),
        supporting_document_ids=("d",),
        question_type="extractive",
        metadata={"x": 1},
    )
    assert Question.from_dict(question.to_dict()) == question
    assert question.to_query().text == "Why?"
    reference = question.to_reference()
    assert reference.answers == ("because",)
    assert reference.relevant_ids == ("d",)
    with pytest.raises(ValueError):
        Question(id="q", text="t", answers=("a", 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Question(id="", text="t")


def test_dataset_metadata() -> None:
    meta = DatasetMetadata(
        name="squad", adapter="SquadDataset", sources=("a.json",), num_documents=3
    )
    data = meta.to_dict()
    assert data["sources"] == ["a.json"]
    assert data["num_documents"] == 3
    with pytest.raises(ValueError):
        DatasetMetadata(name="", adapter="X")
    with pytest.raises(ValueError):
        DatasetMetadata(name="n", adapter="X", num_questions=-1)
