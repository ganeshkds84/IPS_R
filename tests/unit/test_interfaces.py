"""Contract tests for the abstract component interfaces."""

from __future__ import annotations

import pytest

from ips_r.candidates import AbstractCandidateGenerator
from ips_r.components import AbstractComponent
from ips_r.core.exceptions import ComponentError, ConfigurationError
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
from tests.fakes import (
    DOCS,
    QUERY,
    FailingRetriever,
    FakeCandidateGenerator,
    FakeDataset,
    FakeDecisionPolicy,
    FakeEvaluator,
    FakeGenerator,
    FakeReranker,
    FakeRetriever,
)

ABSTRACTS: list[type[AbstractComponent]] = [
    AbstractDataset,
    AbstractRetriever,
    AbstractCandidateGenerator,
    AbstractDecisionPolicy,
    AbstractReranker,
    AbstractGenerator,
    AbstractEvaluator,
    AbstractPipeline,
]


@pytest.mark.parametrize("cls", ABSTRACTS)
def test_interfaces_cannot_be_instantiated(cls: type[AbstractComponent]) -> None:
    with pytest.raises(TypeError):
        cls()


def test_lifecycle_is_idempotent_and_lazy() -> None:
    retriever = FakeRetriever()
    assert not retriever.is_ready
    retriever.index(DOCS)  # lazy setup
    assert retriever.is_ready
    retriever.setup()
    assert retriever.setup_calls == 1
    with retriever:
        pass
    assert not retriever.is_ready
    assert retriever.teardown_calls == 1
    assert "FakeRetriever" in repr(retriever)


def test_component_config_access() -> None:
    retriever = FakeRetriever({"k": 3}, name="r")
    assert retriever.name == "r"
    assert retriever.param("k") == 3
    assert retriever.param("absent", "d") == "d"
    with pytest.raises(ConfigurationError):
        retriever.param("absent")
    with pytest.raises(TypeError):
        retriever.config["k"] = 4  # type: ignore[index]


def test_retriever_contract() -> None:
    retriever = FakeRetriever()
    retriever.index(DOCS)
    results = retriever.retrieve(QUERY, top_k=2)
    assert [r.document.id for r in results] == ["d0", "d1"]
    assert len(retriever.retrieve_batch([QUERY, QUERY], top_k=1)) == 2
    with pytest.raises(ComponentError, match="top_k"):
        retriever.retrieve(QUERY, top_k=0)
    with pytest.raises(ComponentError, match="empty"):
        retriever.index([])
    overshooting = FakeRetriever({"overshoot": 1})
    overshooting.index(DOCS)
    with pytest.raises(ComponentError, match="returned 3 results"):
        overshooting.retrieve(QUERY, top_k=2)


def test_unexpected_errors_are_wrapped() -> None:
    retriever = FailingRetriever(name="broken")
    retriever.index(DOCS)
    with pytest.raises(ComponentError, match=r"\[broken\] retrieve failed") as info:
        retriever.retrieve(QUERY)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_candidate_generator_and_policy() -> None:
    context = [ScoredDocument(d, 1.0) for d in DOCS[:2]]
    candidates = FakeCandidateGenerator().generate(QUERY, context)
    assert [c.id for c in candidates] == ["c0", "c1"]
    decision = FakeDecisionPolicy().decide(QUERY, candidates)
    assert decision.selected_ids == ("c0",)
    with pytest.raises(ComponentError, match="unknown candidate"):
        FakeDecisionPolicy({"force_ids": ["zzz"]}).decide(QUERY, candidates)


def test_reranker_contract() -> None:
    docs = [ScoredDocument(d, float(i)) for i, d in enumerate(DOCS)]
    reranker = FakeReranker()
    assert reranker.rerank(QUERY, []) == []
    assert len(reranker.rerank(QUERY, docs, top_k=2)) == 2
    with pytest.raises(ComponentError, match="not in the input"):
        FakeReranker({"inject_foreign": True}).rerank(QUERY, docs)
    with pytest.raises(ComponentError, match="top_k"):
        reranker.rerank(QUERY, docs, top_k=-1)


def test_generator_contract() -> None:
    context = [ScoredDocument(DOCS[0], 1.0)]
    assert FakeGenerator().generate(QUERY, context).text == "document 0"
    with pytest.raises(ComponentError, match="does not match"):
        FakeGenerator({"query_id": "other"}).generate(QUERY, context)
    with pytest.raises(ComponentError, match="contexts"):
        FakeGenerator().generate_batch([QUERY], [])


def test_evaluator_contract() -> None:
    evaluator = FakeEvaluator()
    preds = [Generation(query_id="q1", text="a")]
    refs = [Reference(query_id="q1", answers=("a",))]
    assert evaluator.evaluate(preds, refs).metrics["accuracy"] == 1.0
    with pytest.raises(ComponentError, match="empty"):
        evaluator.evaluate([], [])
    with pytest.raises(ComponentError, match="references"):
        evaluator.evaluate(preds, [])
    with pytest.raises(ComponentError, match="misaligned"):
        evaluator.evaluate(preds, [Reference(query_id="q2")])


def test_dataset_accessors() -> None:
    dataset = FakeDataset()
    assert len(dataset.queries()) == 1
    assert len(dataset.documents()) == len(DOCS)
    assert dataset.references()[0].query_id == "q1"


def test_data_contract_validation() -> None:
    with pytest.raises(ValueError):
        Query(id="", text="x")
    with pytest.raises(ValueError):
        ScoredDocument(Document(id="d", text=""), score=float("nan"))
    with pytest.raises(ValueError):
        Decision(action="")
    with pytest.raises(ValueError):
        Candidate(id="c", content="", score=float("inf"))
    with pytest.raises(ValueError):
        EvaluationResult(metrics={"m": float("nan")})
    with pytest.raises(TypeError):
        Query(id="q", text="", metadata={"a": 1}).metadata["a"] = 2  # type: ignore[index]
