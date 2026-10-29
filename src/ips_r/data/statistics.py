"""Streaming dataset statistics with JSON and Markdown rendering.

:class:`StatisticsAccumulator` consumes documents, chunks and questions one at a time
(memory grows with the vocabulary and the number of records, never with text size)
and produces an immutable :class:`DatasetStatistics`.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ips_r.config.schema import StatisticsConfig
from ips_r.data.models import Chunk, Document, Question
from ips_r.data.text import compile_pattern, normalize_whitespace


@dataclass(frozen=True, slots=True)
class LengthStats:
    """Summary of a distribution of non-negative lengths."""

    count: int = 0
    total: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: int = 0
    p50: float = 0.0
    p90: float = 0.0
    max: int = 0

    @classmethod
    def from_values(cls, values: array[int]) -> LengthStats:
        """Compute the summary of ``values``."""
        if not values:
            return cls()
        data = np.frombuffer(values, dtype=np.int64)
        return cls(
            count=int(data.size),
            total=int(data.sum()),
            mean=round(float(data.mean()), 3),
            std=round(float(data.std()), 3),
            min=int(data.min()),
            p50=round(float(np.percentile(data, 50)), 3),
            p90=round(float(np.percentile(data, 90)), 3),
            max=int(data.max()),
        )


@dataclass(frozen=True, slots=True)
class VocabularyStats:
    """Vocabulary of the document collection."""

    total_tokens: int = 0
    vocabulary_size: int = 0
    type_token_ratio: float = 0.0
    hapax_legomena: int = 0
    top_terms: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Descriptive statistics of a prepared dataset.

    ``duplicate_ratio`` is the fraction of documents whose whitespace-normalised text
    exactly matches an earlier document (``chunk_duplicate_ratio`` likewise for chunks).
    ``num_pages`` sums the page counts of paginated documents (``None`` if there are none).
    """

    dataset: str
    num_documents: int
    num_questions: int
    num_chunks: int
    num_pages: int | None
    document_chars: LengthStats
    document_tokens: LengthStats
    chunk_chars: LengthStats
    chunk_tokens: LengthStats
    chunks_per_document: LengthStats
    question_tokens: LengthStats
    answers_per_question: LengthStats
    vocabulary: VocabularyStats
    duplicate_documents: int
    duplicate_ratio: float
    duplicate_chunks: int
    chunk_duplicate_ratio: float
    chunk_strategies: Mapping[str, int] = field(default_factory=dict)

    @property
    def average_chunk_size(self) -> float:
        """Mean chunk length in characters."""
        return self.chunk_chars.mean

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data = asdict(self)
        data["average_chunk_size"] = self.average_chunk_size
        data["vocabulary"]["top_terms"] = [list(t) for t in self.vocabulary.top_terms]
        data["chunk_strategies"] = dict(self.chunk_strategies)
        return data

    def to_markdown(self) -> str:
        """Render a human-readable Markdown report."""
        v = self.vocabulary
        lines = [
            f"# Dataset statistics: {self.dataset}",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Documents | {self.num_documents} |",
            f"| Pages | {self.num_pages if self.num_pages is not None else 'n/a'} |",
            f"| Questions | {self.num_questions} |",
            f"| Chunks | {self.num_chunks} |",
            f"| Average chunk size (chars) | {self.average_chunk_size:.1f} |",
            f"| Duplicate documents | {self.duplicate_documents} ({self.duplicate_ratio:.2%}) |",
            f"| Duplicate chunks | {self.duplicate_chunks} ({self.chunk_duplicate_ratio:.2%}) |",
            "",
            "## Length distributions",
            "",
            "| Distribution | Count | Mean | Std | Min | P50 | P90 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, stats in (
            ("Document chars", self.document_chars),
            ("Document tokens", self.document_tokens),
            ("Chunk chars", self.chunk_chars),
            ("Chunk tokens", self.chunk_tokens),
            ("Chunks per document", self.chunks_per_document),
            ("Question tokens", self.question_tokens),
            ("Answers per question", self.answers_per_question),
        ):
            lines.append(
                f"| {label} | {stats.count} | {stats.mean:.1f} | {stats.std:.1f} | {stats.min} "
                f"| {stats.p50:.1f} | {stats.p90:.1f} | {stats.max} |"
            )
        lines += [
            "",
            "## Vocabulary",
            "",
            f"- Total tokens: {v.total_tokens}",
            f"- Vocabulary size: {v.vocabulary_size}",
            f"- Type/token ratio: {v.type_token_ratio:.4f}",
            f"- Hapax legomena: {v.hapax_legomena}",
        ]
        if v.top_terms:
            lines += ["", "| Term | Count |", "| --- | ---: |"]
            lines += [f"| `{term}` | {count} |" for term, count in v.top_terms]
        if self.chunk_strategies:
            lines += ["", "## Chunking", ""]
            lines += [
                f"- `{name}`: {count} chunks" for name, count in self.chunk_strategies.items()
            ]
        return "\n".join(lines) + "\n"


def _fingerprint(text: str) -> bytes:
    """Return a compact digest of whitespace-normalised, case-folded text."""
    return hashlib.blake2b(
        normalize_whitespace(text).casefold().encode("utf-8", "surrogatepass"), digest_size=12
    ).digest()


class StatisticsAccumulator:
    """Incrementally computes :class:`DatasetStatistics`.

    Args:
        dataset: Dataset name.
        settings: Tokenisation and reporting options.
    """

    def __init__(self, dataset: str, settings: StatisticsConfig | None = None) -> None:
        self._dataset = dataset
        self._s = settings or StatisticsConfig()
        if self._s.top_k_terms < 0:
            raise ValueError("statistics.top_k_terms must be >= 0")
        self._word = compile_pattern(self._s.word_pattern)
        self._vocab: Counter[str] = Counter()
        self._doc_chars: array[int] = array("q")
        self._doc_tokens: array[int] = array("q")
        self._chunk_chars: array[int] = array("q")
        self._chunk_tokens: array[int] = array("q")
        self._question_tokens: array[int] = array("q")
        self._answers: array[int] = array("q")
        self._chunks_per_doc: Counter[str] = Counter()
        self._strategies: Counter[str] = Counter()
        self._doc_fingerprints: set[bytes] = set()
        self._chunk_fingerprints: set[bytes] = set()
        self._duplicate_docs = 0
        self._duplicate_chunks = 0
        self._pages = 0
        self._has_pages = False

    def _words(self, text: str) -> list[str]:
        """Tokenise ``text`` with the configured word pattern."""
        words = self._word.findall(text)
        return [w.lower() for w in words] if self._s.lowercase else words

    def add_document(self, document: Document) -> None:
        """Account for one document."""
        words = self._words(document.text)
        self._vocab.update(words)
        self._doc_chars.append(len(document.text))
        self._doc_tokens.append(len(words))
        self._chunks_per_doc.setdefault(document.id, 0)
        if document.num_pages is not None:
            self._pages += document.num_pages
            self._has_pages = True
        fingerprint = _fingerprint(document.text)
        if fingerprint in self._doc_fingerprints:
            self._duplicate_docs += 1
        self._doc_fingerprints.add(fingerprint)

    def add_chunk(self, chunk: Chunk) -> None:
        """Account for one chunk."""
        self._chunk_chars.append(len(chunk.text))
        self._chunk_tokens.append(len(self._word.findall(chunk.text)))
        self._chunks_per_doc[chunk.document_id] += 1
        self._strategies[chunk.strategy] += 1
        fingerprint = _fingerprint(chunk.text)
        if fingerprint in self._chunk_fingerprints:
            self._duplicate_chunks += 1
        self._chunk_fingerprints.add(fingerprint)

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Account for several chunks."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def add_question(self, question: Question) -> None:
        """Account for one question."""
        self._question_tokens.append(len(self._word.findall(question.text)))
        self._answers.append(len(question.answers))

    def result(self) -> DatasetStatistics:
        """Return the statistics accumulated so far."""
        num_docs = len(self._doc_chars)
        num_chunks = len(self._chunk_chars)
        total_tokens = sum(self._vocab.values())
        top = sorted(self._vocab.items(), key=lambda item: (-item[1], item[0]))[
            : self._s.top_k_terms
        ]
        per_doc = array("q", self._chunks_per_doc.values()) if num_chunks else array("q")
        return DatasetStatistics(
            dataset=self._dataset,
            num_documents=num_docs,
            num_questions=len(self._question_tokens),
            num_chunks=num_chunks,
            num_pages=self._pages if self._has_pages else None,
            document_chars=LengthStats.from_values(self._doc_chars),
            document_tokens=LengthStats.from_values(self._doc_tokens),
            chunk_chars=LengthStats.from_values(self._chunk_chars),
            chunk_tokens=LengthStats.from_values(self._chunk_tokens),
            chunks_per_document=LengthStats.from_values(per_doc),
            question_tokens=LengthStats.from_values(self._question_tokens),
            answers_per_question=LengthStats.from_values(self._answers),
            vocabulary=VocabularyStats(
                total_tokens=total_tokens,
                vocabulary_size=len(self._vocab),
                type_token_ratio=round(len(self._vocab) / total_tokens, 6) if total_tokens else 0.0,
                hapax_legomena=sum(1 for count in self._vocab.values() if count == 1),
                top_terms=tuple(top),
            ),
            duplicate_documents=self._duplicate_docs,
            duplicate_ratio=_ratio(self._duplicate_docs, num_docs),
            duplicate_chunks=self._duplicate_chunks,
            chunk_duplicate_ratio=_ratio(self._duplicate_chunks, num_chunks),
            chunk_strategies=dict(sorted(self._strategies.items())),
        )


def _ratio(part: int, whole: int) -> float:
    """Return ``part / whole`` rounded, or 0 for an empty whole."""
    return round(part / whole, 6) if whole else 0.0
