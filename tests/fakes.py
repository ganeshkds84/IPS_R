"""Minimal in-memory implementations of every interface, used only by tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ips_r.candidates import AbstractCandidateGenerator
from ips_r.core.types import (
    Candidate,
    Decision,
    Document,
    EvaluationResult,
    Generation,
    Query,
    Reference,
    ScoredDocument,
)
from ips_r.data import AbstractDataset
from ips_r.evaluation import AbstractEvaluator
from ips_r.generation import AbstractGenerator
from ips_r.pipeline import AbstractPipeline
from ips_r.policy import AbstractDecisionPolicy
from ips_r.reranking import AbstractReranker
from ips_r.retrieval import AbstractRetriever

DOCS = [Document(id=f"d{i}", text=f"document {i}") for i in range(5)]
QUERY = Query(id="q1", text="find document")


class FakeDataset(AbstractDataset):
    def _load_queries(self) -> list[Query]:
        return [QUERY]

    def _load_documents(self) -> list[Document]:
        return list(DOCS)

    def _load_references(self) -> list[Reference]:
        return [Reference(query_id=QUERY.id, answers=("document 0",))]


class FakeRetriever(AbstractRetriever):
    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self.docs: list[Document] = []
        self.setup_calls = 0
        self.teardown_calls = 0

    def _setup(self) -> None:
        self.setup_calls += 1

    def _teardown(self) -> None:
        self.teardown_calls += 1

    def _index(self, documents: Sequence[Document]) -> None:
        self.docs = list(documents)

    def _retrieve(self, query: Query, top_k: int) -> list[ScoredDocument]:
        overshoot = int(self.param("overshoot", 0))
        n = top_k + overshoot
        return [
            ScoredDocument(d, score=float(len(self.docs) - i)) for i, d in enumerate(self.docs[:n])
        ]


class FailingRetriever(FakeRetriever):
    def _retrieve(self, query: Query, top_k: int) -> list[ScoredDocument]:
        raise RuntimeError("index corrupted")


class FakeCandidateGenerator(AbstractCandidateGenerator):
    def _generate(self, query: Query, context: Sequence[ScoredDocument]) -> list[Candidate]:
        return [
            Candidate(id=f"c{i}", content=item.document.text, source_ids=(item.document.id,))
            for i, item in enumerate(context)
        ]


class FakeDecisionPolicy(AbstractDecisionPolicy):
    def _decide(self, query: Query, candidates: Sequence[Candidate]) -> Decision:
        selected = self.param("force_ids", None)
        ids = tuple(selected) if selected else tuple(c.id for c in candidates[:1])
        return Decision(action="answer", confidence=1.0, selected_ids=ids)


class FakeReranker(AbstractReranker):
    def _rerank(
        self, query: Query, documents: Sequence[ScoredDocument], top_k: int | None
    ) -> list[ScoredDocument]:
        if self.param("inject_foreign", False):
            return [ScoredDocument(Document(id="alien", text=""), score=0.0)]
        ranked = sorted(documents, key=lambda item: item.score)
        return ranked[:top_k] if top_k is not None else ranked


class FakeGenerator(AbstractGenerator):
    def _generate(self, query: Query, context: Sequence[ScoredDocument]) -> Generation:
        query_id = str(self.param("query_id", query.id))
        return Generation(query_id=query_id, text=context[0].document.text if context else "")


class FakeEvaluator(AbstractEvaluator):
    def _evaluate(
        self, predictions: Sequence[Generation], references: Sequence[Reference]
    ) -> EvaluationResult:
        hits = [float(p.text in r.answers) for p, r in zip(predictions, references, strict=True)]
        return EvaluationResult(metrics={"accuracy": sum(hits) / len(hits)})


class FakePipeline(AbstractPipeline):
    def _run(self) -> dict[str, Any]:
        c = self.components
        assert c.dataset and c.retriever and c.generator and c.evaluator
        c.retriever.index(c.dataset.documents())
        queries = c.dataset.queries()
        contexts = c.retriever.retrieve_batch(queries, top_k=2)
        predictions = c.generator.generate_batch(queries, contexts)
        result = c.evaluator.evaluate(predictions, c.dataset.references())
        return result.to_dict()
