"""Tests for streaming dataset statistics."""

from __future__ import annotations

from array import array

import pytest

from ips_r.config.schema import StatisticsConfig
from ips_r.data.models import Chunk, Document, Question
from ips_r.data.statistics import LengthStats, StatisticsAccumulator


def _chunk(doc: str, index: int, text: str) -> Chunk:
    return Chunk(
        id=f"{doc}::{index}",
        document_id=doc,
        text=text,
        index=index,
        start=0,
        end=len(text),
        strategy="token",
    )


def test_length_stats() -> None:
    stats = LengthStats.from_values(array("q", [1, 2, 3, 10]))
    assert (stats.count, stats.total, stats.min, stats.max) == (4, 16, 1, 10)
    assert stats.mean == 4.0
    assert stats.p50 == 2.5
    assert LengthStats.from_values(array("q")) == LengthStats()


def test_accumulator_end_to_end() -> None:
    acc = StatisticsAccumulator("ds", StatisticsConfig(top_k_terms=2))
    acc.add_document(Document(id="a", text="The cat sat.", num_pages=2))
    acc.add_document(Document(id="b", text="the  CAT sat."))
    acc.add_document(Document(id="c", text="A dog ran."))
    acc.add_chunks([_chunk("a", 0, "The cat"), _chunk("a", 1, "sat."), _chunk("c", 0, "The cat")])
    acc.add_question(Question(id="q", text="Who sat?", answers=("cat", "the cat")))
    result = acc.result()
    assert (result.num_documents, result.num_chunks, result.num_questions) == (3, 3, 1)
    assert result.num_pages == 2
    assert result.duplicate_documents == 1
    assert result.duplicate_ratio == pytest.approx(1 / 3, rel=1e-4)
    assert result.duplicate_chunks == 1
    assert result.vocabulary.total_tokens == 9
    assert result.vocabulary.vocabulary_size == 6  # the, cat, sat, a, dog, ran
    assert result.vocabulary.top_terms == (("cat", 2), ("sat", 2))
    assert result.vocabulary.hapax_legomena == 3
    assert result.chunks_per_document.max == 2
    assert result.chunks_per_document.min == 0
    assert result.answers_per_question.mean == 2.0
    assert result.average_chunk_size == result.chunk_chars.mean
    assert result.chunk_strategies == {"token": 3}
    data = result.to_dict()
    assert data["vocabulary"]["top_terms"] == [["cat", 2], ["sat", 2]]
    markdown = result.to_markdown()
    assert "| Documents | 3 |" in markdown
    assert "`token`: 3 chunks" in markdown


def test_empty_accumulator_and_case_sensitivity() -> None:
    empty = StatisticsAccumulator("empty").result()
    assert empty.num_pages is None
    assert empty.duplicate_ratio == 0.0
    assert empty.vocabulary.type_token_ratio == 0.0
    assert "n/a" in empty.to_markdown()
    acc = StatisticsAccumulator("ds", StatisticsConfig(lowercase=False))
    acc.add_document(Document(id="a", text="Cat cat"))
    assert acc.result().vocabulary.vocabulary_size == 2
    with pytest.raises(ValueError):
        StatisticsAccumulator("ds", StatisticsConfig(top_k_terms=-1))
