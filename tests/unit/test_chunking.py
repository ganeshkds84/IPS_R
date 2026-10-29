"""Tests for chunking strategies and the chunker contract."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import pytest

from ips_r.core.exceptions import ComponentError, ConfigurationError
from ips_r.data.chunking import (
    CHUNKERS,
    AbstractChunker,
    AbstractSemanticChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    SlidingWindowChunker,
    Span,
    TokenChunker,
    make_chunk_id,
    merge_pieces,
)
from ips_r.data.models import Chunk, Document, ElementDraft, ElementType

TEXT = (
    "Retrieval augmented generation is popular. It grounds answers in documents.\n\n"
    "Query reformulation rewrites the query. Intent must be preserved during rewriting. "
    "Evaluation uses several benchmarks, e.g. HotpotQA and SQuAD."
)


def _assert_contract(document: Document, chunks: Sequence[Chunk]) -> None:
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text == document.text[chunk.start : chunk.end]
        assert chunk.text == chunk.text.strip()
        assert chunk.id == make_chunk_id(document.id, chunk.index)
        assert chunk.document_id == document.id


def _covered(document: Document, chunks: Sequence[Chunk]) -> str:
    """Concatenate non-whitespace characters covered by chunks (to check nothing is lost)."""
    covered = [False] * len(document.text)
    for chunk in chunks:
        for i in range(chunk.start, chunk.end):
            covered[i] = True
    return "".join(
        ch for ch, flag in zip(document.text, covered, strict=True) if flag and not ch.isspace()
    )


@pytest.mark.parametrize(
    ("registry_name", "params"),
    [
        ("recursive_character", {"chunk_size": 60, "chunk_overlap": 20}),
        ("sentence", {"sentences_per_chunk": 2, "sentence_overlap": 1}),
        ("token", {"chunk_size": 10, "chunk_overlap": 3}),
        ("sliding_window", {"window_size": 50, "stride": 30}),
        ("sliding_window", {"window_size": 50, "stride": 30, "respect_word_boundaries": False}),
    ],
)
def test_every_strategy_honours_contract_and_covers_text(
    registry_name: str, params: dict[str, Any]
) -> None:
    document = Document(id="doc", text=TEXT)
    chunks = CHUNKERS.create(registry_name, params, name=registry_name).chunk(document)
    assert len(chunks) > 1
    _assert_contract(document, chunks)
    assert {c.strategy for c in chunks} == {registry_name}
    assert _covered(document, chunks) == "".join(ch for ch in TEXT if not ch.isspace())


def test_recursive_character_respects_size_and_overlap() -> None:
    chunker = RecursiveCharacterChunker(
        {"chunk_size": 60, "chunk_overlap": 20, "separators": [" "]}
    )
    chunks = chunker.chunk(Document(id="d", text=TEXT))
    assert all(c.num_chars <= 60 for c in chunks)
    overlaps = [a.end - b.start for a, b in itertools.pairwise(chunks)]
    assert all(0 < overlap <= 20 for overlap in overlaps)


def test_recursive_character_hard_cuts_long_words() -> None:
    chunker = RecursiveCharacterChunker(
        {"chunk_size": 4, "chunk_overlap": 0, "separators": [" ", ""]}
    )
    chunks = chunker.chunk(Document(id="d", text="abcdefghij kl"))
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij", "kl"]
    no_fallback = RecursiveCharacterChunker(
        {"chunk_size": 4, "chunk_overlap": 0, "separators": ["|"]}
    )
    assert [c.text for c in no_fallback.chunk(Document(id="d", text="abcdefgh"))] == ["abcdefgh"]


def test_merge_pieces() -> None:
    pieces: list[Span] = [(0, 3), (3, 6), (6, 9), (9, 20)]
    # (6, 9) alone would be contained in (3, 9): overlap is dropped when the next piece
    # cannot fit, so the oversized last piece starts a fresh window.
    assert merge_pieces(pieces, size=6, overlap=3) == [(0, 6), (3, 9), (9, 20)]
    assert merge_pieces(pieces, size=9, overlap=0) == [(0, 9), (9, 20)]
    assert merge_pieces([], size=5, overlap=0) == []


def test_sentence_chunker_max_chars_and_groups() -> None:
    chunker = SentenceChunker(
        {"sentences_per_chunk": 3, "sentence_overlap": 0, "max_chunk_chars": 80}
    )
    chunks = chunker.chunk(Document(id="d", text=TEXT))
    assert all(c.num_chars <= 80 or c.text.count(". ") == 0 for c in chunks)
    assert (
        len(
            SentenceChunker({"sentences_per_chunk": 10, "sentence_overlap": 0}).chunk(
                Document(id="d", text=TEXT)
            )
        )
        == 1
    )


def test_token_chunker_counts_tokens() -> None:
    chunker = TokenChunker({"chunk_size": 4, "chunk_overlap": 0, "token_pattern": r"\w+"})
    chunks = chunker.chunk(Document(id="d", text="one two three four five six"))
    assert [c.text for c in chunks] == ["one two three four", "five six"]


def test_sliding_window_snaps_to_words() -> None:
    chunker = SlidingWindowChunker({"window_size": 8, "stride": 6})
    chunks = chunker.chunk(Document(id="d", text="alpha beta gamma delta"))
    assert all(" " not in c.text or c.text.split(" ")[0] for c in chunks)
    assert chunks[0].text == "alpha"
    long_word = SlidingWindowChunker({"window_size": 4, "stride": 2}).chunk(
        Document(id="d", text="abcdefgh")
    )
    assert long_word[0].text == "abcd"


def test_chunks_carry_pages_and_sections() -> None:
    document = Document.from_elements(
        "paper",
        [
            ElementDraft(ElementType.HEADING, "Intro", page=1, section="Intro", level=1),
            ElementDraft(ElementType.PARAGRAPH, "First page text.", page=1, section="Intro"),
            ElementDraft(ElementType.PARAGRAPH, "Second page text.", page=2, section="Intro"),
        ],
        num_pages=2,
    )
    chunks = RecursiveCharacterChunker({"chunk_size": 1000, "chunk_overlap": 0}).chunk(document)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end, chunks[0].section) == (1, 2, "Intro")
    plain = RecursiveCharacterChunker().chunk(Document(id="p", text="no elements"))
    assert (plain[0].page_start, plain[0].section) == (None, None)


def test_empty_documents_and_min_chunk_chars() -> None:
    chunker = TokenChunker({"chunk_size": 1, "chunk_overlap": 0, "min_chunk_chars": 3})
    assert chunker.chunk(Document(id="d", text="   ")) == []
    assert [c.text for c in chunker.chunk(Document(id="d", text="a bb ccc dddd"))] == [
        "ccc",
        "dddd",
    ]
    docs = [Document(id="a", text="x y"), Document(id="b", text="z")]
    assert len(list(TokenChunker({"chunk_size": 1, "chunk_overlap": 0}).chunk_documents(docs))) == 3


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        (RecursiveCharacterChunker, {"chunk_size": 10, "chunk_overlap": 10}),
        (RecursiveCharacterChunker, {"separators": []}),
        (RecursiveCharacterChunker, {"chunk_size": 0}),
        (SentenceChunker, {"sentences_per_chunk": 2, "sentence_overlap": 2}),
        (TokenChunker, {"token_pattern": "("}),
        (TokenChunker, {"chunk_size": "10"}),
        (SlidingWindowChunker, {"window_size": 5, "stride": 6}),
        (SlidingWindowChunker, {"respect_word_boundaries": "yes"}),
    ],
)
def test_invalid_parameters_fail_fast(cls: type[AbstractChunker], params: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        cls(params)


class _BrokenChunker(AbstractChunker):
    strategy = "broken"

    def _split(self, text: str) -> Sequence[Span]:
        return [(0, len(text) + 5)]


def test_invalid_spans_are_rejected() -> None:
    with pytest.raises(ComponentError, match="invalid span"):
        _BrokenChunker().chunk(Document(id="d", text="abc"))


class _FakeSemantic(AbstractSemanticChunker):
    def __init__(self, scores: list[float], config: Any = None) -> None:
        super().__init__(config)
        self._scores = scores

    def _similarities(self, sentences: Sequence[str]) -> Sequence[float]:
        return self._scores[: max(len(sentences) - 1, 0)]


def test_semantic_interface_cuts_on_low_similarity() -> None:
    text = "A one. A two. B one. B two. B three."
    chunks = _FakeSemantic([0.9, 0.1, 0.9, 0.9], {"similarity_threshold": 0.5}).chunk(
        Document(id="d", text=text)
    )
    assert [c.text for c in chunks] == ["A one. A two.", "B one. B two. B three."]
    capped = _FakeSemantic([0.9] * 4, {"max_sentences": 2}).chunk(Document(id="d", text=text))
    assert [c.text for c in capped] == ["A one. A two.", "B one. B two.", "B three."]
    assert _FakeSemantic([]).chunk(Document(id="d", text="Single sentence.")) != []
    with pytest.raises(ComponentError, match="scores"):
        _FakeSemantic([0.1]).chunk(Document(id="d", text=text))
    with pytest.raises(ConfigurationError):
        _FakeSemantic([], {"min_sentences": 5, "max_sentences": 2})


def test_semantic_placeholder_explains_itself() -> None:
    with pytest.raises(Exception, match="placeholder"):
        CHUNKERS.create("semantic", {})
    assert set(CHUNKERS.names()) >= {
        "recursive_character",
        "sentence",
        "token",
        "sliding_window",
        "semantic",
    }
