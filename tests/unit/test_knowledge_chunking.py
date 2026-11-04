"""Tests for chunk overlap validation and edge cases of KnowledgeChunker (Part D)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

from ips_r.config.schema import ChunkProcessingConfig
from ips_r.data.chunking.base import AbstractChunker, Span
from ips_r.data.models import Document
from ips_r.knowledge import ChunkKind, KnowledgeChunker, TextUnit, TrackedText

TEXT = "abcdefghij" * 10  # 100 chars, no whitespace: every span is kept verbatim


class FixedSpans(AbstractChunker):
    """Returns the spans given in ``params.spans``."""

    strategy: ClassVar[str] = "fixed"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._spans = [tuple(span) for span in self.param("spans")]

    def _split(self, text: str) -> Sequence[Span]:
        return [(start, end) for start, end in self._spans]


def _issues(spans: list[tuple[int, int]], ratio: float = 0.5) -> dict[str, int]:
    chunker = KnowledgeChunker(
        ChunkProcessingConfig(max_overlap_ratio=ratio, deduplicate="off"),
        FixedSpans({"spans": spans}),
    )
    unit = TextUnit("u", ChunkKind.TEXT, "s", 0, TrackedText.original(TEXT))
    chunker.chunk(Document(id="d", text=TEXT), [unit], [], source_hash="h", fingerprint="f")
    return dict(chunker.validation.counts)


def test_regular_overlap_is_accepted() -> None:
    assert _issues([(0, 40), (30, 70), (60, 100)]) == {}


def test_short_tail_window_is_not_excessive_overlap() -> None:
    # Regression: the ratio used to be relative to the (short) later chunk only, so a
    # normal 10-char overlap into a 15-char tail window was flagged as excessive.
    assert _issues([(0, 40), (30, 70), (60, 75)]) == {}


def test_excessive_overlap_is_relative_to_the_longer_neighbour() -> None:
    assert _issues([(0, 40), (10, 50)]) == {"excessive_overlap": 1}
    assert _issues([(0, 40), (10, 50)], ratio=0.8) == {}


def test_contained_and_gap_issues() -> None:
    assert _issues([(0, 40), (10, 30)]) == {"contained": 1}
    assert _issues([(0, 40), (50, 90)]) == {"gap": 1}


def test_whitespace_only_units_produce_no_chunks() -> None:
    chunker = KnowledgeChunker()
    assert not chunker.has_base_chunker
    unit = TextUnit("u", ChunkKind.TEXT, "s", 0, TrackedText.original("   \n "))
    document = Document(id="d", text="x   \n ")
    assert chunker.chunk(document, [unit], [], source_hash="h", fingerprint="f") == []


@pytest.mark.parametrize("mode", ["mark", "drop", "off"])
def test_dedup_modes(mode: str) -> None:
    chunker = KnowledgeChunker(ChunkProcessingConfig(deduplicate=mode))
    units = [
        TextUnit(f"u{i}", ChunkKind.TEXT, "s", i, TrackedText.original("Same  text."))
        for i in range(2)
    ]
    document = Document(id="d", text="Same  text.")
    chunks = chunker.chunk(document, units, [], source_hash="h", fingerprint="f")
    expected = {"mark": 2, "drop": 1, "off": 2}[mode]
    assert len(chunks) == expected
    if mode == "mark":
        assert chunks[1].duplicate_of == chunks[0].id
    assert chunker.validation.duplicates == (0 if mode == "off" else 1)
