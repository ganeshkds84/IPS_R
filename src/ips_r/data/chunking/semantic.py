"""Semantic chunking: interface only (Milestone 1).

Semantic chunking places boundaries where adjacent sentences are dissimilar. The
boundary *placement* is implemented here; the *similarity scoring* requires an
embedding model, which is out of scope for the dataset layer. A later milestone
implements :meth:`AbstractSemanticChunker._similarities` and registers the result.

``configs/chunker/semantic.yaml`` selects :class:`SemanticChunkerPlaceholder`, which
fails at construction (so ``--dry-run`` reports it) with an explicit message.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from ips_r.core.exceptions import ConfigurationError
from ips_r.data.chunking.base import CHUNKERS, AbstractChunker, Span
from ips_r.data.text import DEFAULT_ABBREVIATIONS, sentence_spans


class AbstractSemanticChunker(AbstractChunker):
    """Sentence-level chunker cutting where adjacent-sentence similarity is low.

    Parameters:
        similarity_threshold: Cut between sentences whose similarity is below this value.
        min_sentences: Never cut a chunk shorter than this many sentences.
        max_sentences: Always cut once a chunk has this many sentences.
        abbreviations: Tokens whose trailing period does not end a sentence.
    """

    strategy: ClassVar[str] = "semantic"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._threshold = self.float_param("similarity_threshold", 0.5, minimum=-1.0, maximum=1.0)
        self._min_sentences = self.int_param("min_sentences", 1, minimum=1)
        self._max_sentences = self.int_param("max_sentences", 20, minimum=1)
        self._abbreviations = self.str_list_param("abbreviations", list(DEFAULT_ABBREVIATIONS))
        if self._min_sentences > self._max_sentences:
            raise ConfigurationError(
                f"chunker '{self.name}': min_sentences must not exceed max_sentences"
            )

    def _split(self, text: str) -> list[Span]:
        sentences = sentence_spans(text, self._abbreviations)
        if not sentences:
            return []
        scores = list(self._similarities([text[s:e] for s, e in sentences]))
        if len(scores) != len(sentences) - 1:
            raise ValueError(
                f"_similarities returned {len(scores)} scores for {len(sentences)} sentences"
            )
        chunks: list[Span] = []
        first = 0
        for i, score in enumerate(scores):
            size = i + 1 - first
            if size >= self._max_sentences or (
                size >= self._min_sentences and score < self._threshold
            ):
                chunks.append((sentences[first][0], sentences[i][1]))
                first = i + 1
        chunks.append((sentences[first][0], sentences[-1][1]))
        return chunks

    @abstractmethod
    def _similarities(self, sentences: Sequence[str]) -> Sequence[float]:
        """Return the similarity of each adjacent sentence pair (``len(sentences) - 1`` values)."""
        raise NotImplementedError


@CHUNKERS.register("semantic")
class SemanticChunkerPlaceholder(AbstractSemanticChunker):
    """Registered placeholder that explains semantic chunking is not available yet.

    Raises:
        ConfigurationError: Always, on construction.
    """

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        raise ConfigurationError(
            "chunker 'semantic' is an interface placeholder: similarity scoring requires an "
            "embedding model, which the dataset layer does not provide. Use "
            "recursive_character, sentence, token or sliding_window."
        )

    def _similarities(self, sentences: Sequence[str]) -> Sequence[float]:  # pragma: no cover
        raise NotImplementedError
