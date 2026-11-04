"""Tests for the embedding and index interfaces (Parts E and F), driven by in-memory doubles."""

from __future__ import annotations

import math

import pytest

from ips_r.core.exceptions import ComponentError
from ips_r.knowledge import (
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingRequest,
    EmbeddingResult,
    IndexArtifact,
    KnowledgeChunk,
    KnowledgeProcessor,
    RetrievalArtifact,
    ScoredId,
    VectorRecord,
)
from tests.knowledge_factories import paper
from tests.knowledge_fakes import (
    DictCache,
    HashEmbedding,
    MemorySparseIndexer,
    MemoryVectorStore,
    SumHybrid,
)


@pytest.fixture(scope="module")
def chunks() -> tuple[KnowledgeChunk, ...]:
    return KnowledgeProcessor().process(paper()).chunks


# --------------------------------------------------------------------------- #
# Embedding data contracts
# --------------------------------------------------------------------------- #
def test_embedding_request_hashes_text_and_freezes_metadata() -> None:
    request = EmbeddingRequest("q1", "hello", kind="query", metadata={"a": 1})
    assert request.content_hash.startswith("sha256:")
    assert request.content_hash == EmbeddingRequest("q2", "hello").content_hash
    with pytest.raises(TypeError):
        request.metadata["a"] = 2  # type: ignore[index]  # read-only mapping proxy


@pytest.mark.parametrize(
    ("request_id", "text", "kind", "message"),
    [
        ("", "x", "document", "non-empty"),
        ("a", "", "document", "non-empty"),
        ("a", "x", "passage", "kind"),
    ],
)
def test_embedding_request_validation(request_id: str, text: str, kind: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingRequest(request_id, text, kind=kind)


def test_embedding_request_from_chunk_reuses_chunk_hash(chunks: tuple[KnowledgeChunk, ...]) -> None:
    request = EmbeddingRequest.from_chunk(chunks[0])
    assert (request.id, request.text, request.kind) == (chunks[0].id, chunks[0].text, "document")
    assert request.content_hash == chunks[0].content_hash


def test_embedding_batches_are_ordered_unique_and_deterministic() -> None:
    requests = [EmbeddingRequest(f"r{i}", f"text {i}") for i in range(5)]
    batches = list(EmbeddingBatch.batches(requests, 2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert batches[0].batch_id == EmbeddingBatch(tuple(requests[:2])).batch_id
    assert batches[0].batch_id != batches[1].batch_id
    with pytest.raises(ValueError, match="batch_size"):
        list(EmbeddingBatch.batches(requests, 0))
    with pytest.raises(ValueError, match="empty"):
        EmbeddingBatch(())
    with pytest.raises(ValueError, match="unique"):
        EmbeddingBatch((requests[0], requests[0]))


def test_embedding_result_validation() -> None:
    result = EmbeddingResult("r", [1, 2], "m", "h")  # type: ignore[arg-type]  # coerced to floats
    assert result.vector == (1.0, 2.0) and result.dimension == 2
    for bad in ((), (math.inf,)):
        with pytest.raises(ValueError, match="finite"):
            EmbeddingResult("r", bad, "m", "h")


def test_cache_key_depends_on_model_kind_and_content() -> None:
    key = EmbeddingCache.key("m", "document", "h")
    assert key == EmbeddingCache.key("m", "document", "h")
    assert (
        len({key, EmbeddingCache.key("n", "document", "h"), EmbeddingCache.key("m", "query", "h")})
        == 3
    )


# --------------------------------------------------------------------------- #
# AbstractEmbeddingModel template method
# --------------------------------------------------------------------------- #
def test_embed_serves_cache_hits_and_embeds_only_misses() -> None:
    cache = DictCache()
    model = HashEmbedding({"batch_size": 8}, name="hash", cache=cache)
    assert model.model_id == "hash"
    first = EmbeddingBatch((EmbeddingRequest("a", "alpha"), EmbeddingRequest("b", "beta")))
    results = model.embed(first)
    assert [r.request_id for r in results] == ["a", "b"]
    assert not any(r.cached for r in results)
    assert all(r.model == "hash" and r.dimension == 4 for r in results)
    second = EmbeddingBatch((EmbeddingRequest("c", "gamma"), EmbeddingRequest("b2", "beta")))
    again = model.embed(second)
    assert [r.cached for r in again] == [False, True]
    assert again[1].vector == results[1].vector
    assert model.calls == [2, 1]
    assert len(cache.store) == 3


def test_embed_without_cache_and_all_hits_skip_the_model() -> None:
    cache = DictCache()
    model = HashEmbedding(name="hash", cache=cache)
    batch = EmbeddingBatch((EmbeddingRequest("a", "alpha"),))
    model.embed(batch)
    model.embed(batch)
    assert model.calls == [1]
    plain = HashEmbedding()
    plain.embed(batch)
    plain.embed(batch)
    assert plain.calls == [1, 1]


def test_embed_normalises_vectors() -> None:
    model = HashEmbedding({"normalize": True, "dimension": 3})
    (result,) = model.embed(EmbeddingBatch((EmbeddingRequest("a", "alpha"),)))
    assert math.isclose(math.sqrt(sum(v * v for v in result.vector)), 1.0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mode": "short"}, "vectors for"),
        ({"mode": "wrong_dim"}, "dimension"),
        ({"mode": "nan"}, "non-finite"),
        ({"mode": "zero", "normalize": True}, "zero vector"),
    ],
)
def test_embed_rejects_invalid_model_output(params: dict[str, object], message: str) -> None:
    model = HashEmbedding(params)
    with pytest.raises(ComponentError, match=message):
        model.embed(EmbeddingBatch((EmbeddingRequest("a", "x"), EmbeddingRequest("b", "y"))))


def test_embed_chunks_batches_by_batch_size(chunks: tuple[KnowledgeChunk, ...]) -> None:
    model = HashEmbedding({"batch_size": 3})
    results = model.embed_chunks(chunks)
    assert [r.request_id for r in results] == [c.id for c in chunks]
    assert sum(model.calls) == len(chunks) and max(model.calls) == 3


# --------------------------------------------------------------------------- #
# Index data contracts
# --------------------------------------------------------------------------- #
def test_vector_record_and_scored_id_validation() -> None:
    record = VectorRecord("a", (1, 2), {"k": "v"})  # ints are coerced to floats
    assert record.vector == (1.0, 2.0)
    with pytest.raises(ValueError, match="id"):
        VectorRecord("", (1.0,))
    with pytest.raises(ValueError, match="finite"):
        VectorRecord("a", (math.nan,))
    with pytest.raises(ValueError, match="finite"):
        ScoredId("a", math.inf)


def test_index_artifact_hash_and_staleness() -> None:
    artifact = IndexArtifact("v", "vector", "Mem", 3, "sha256:abc", "sha256:cfg", dimension=4)
    assert artifact.artifact_hash == IndexArtifact(**artifact.to_dict()).artifact_hash
    assert artifact.to_dict()["metadata"] == {}
    assert not artifact.is_stale("sha256:abc")
    assert artifact.is_stale("sha256:other")
    with pytest.raises(ValueError, match="kind"):
        IndexArtifact("v", "graph", "Mem", 0, "h", "c")
    with pytest.raises(ValueError, match="num_items"):
        IndexArtifact("v", "vector", "Mem", -1, "h", "c")
    with pytest.raises(ValueError, match="dimension"):
        IndexArtifact("v", "vector", "Mem", 0, "h", "c", dimension=0)


def test_retrieval_artifact_validation() -> None:
    artifact = RetrievalArtifact("q", (ScoredId("a", 2.0), ScoredId("b", 1.0)), "sha256:i")
    assert artifact.to_dict()["results"] == [{"id": "a", "score": 2.0}, {"id": "b", "score": 1.0}]
    with pytest.raises(ValueError, match="unique"):
        RetrievalArtifact("q", (ScoredId("a", 2.0), ScoredId("a", 1.0)), "h")
    with pytest.raises(ValueError, match="descending"):
        RetrievalArtifact("q", (ScoredId("a", 1.0), ScoredId("b", 2.0)), "h")


# --------------------------------------------------------------------------- #
# Index template methods
# --------------------------------------------------------------------------- #
def test_vector_store_contract() -> None:
    store = MemoryVectorStore(name="vs")
    store.upsert([])
    store.delete([])
    store.upsert([VectorRecord("a", (1.0, 0.0), {"doc": "x"}), VectorRecord("b", (0.0, 1.0))])
    assert store.count() == 2
    assert [s.id for s in store.search((1.0, 0.0), top_k=1)] == ["a"]
    assert [s.id for s in store.search((1.0, 1.0), filters={"doc": "x"})] == ["a"]
    store.delete(["a", "missing"])
    assert store.count() == 1
    artifact = store.artifact("sha256:k", location="mem://vs")
    assert (artifact.kind, artifact.num_items, artifact.dimension) == ("vector", 1, 2)
    assert artifact.backend == "MemoryVectorStore" and artifact.location == "mem://vs"


def test_vector_store_rejects_invalid_input_and_rankings() -> None:
    store = MemoryVectorStore()
    with pytest.raises(ComponentError, match="duplicate"):
        store.upsert([VectorRecord("a", (1.0,)), VectorRecord("a", (2.0,))])
    with pytest.raises(ComponentError, match="dimensions"):
        store.upsert([VectorRecord("a", (1.0,)), VectorRecord("b", (1.0, 2.0))])
    with pytest.raises(ComponentError, match="top_k"):
        store.search((1.0,), top_k=0)
    unsorted = MemoryVectorStore({"unsorted": True})
    unsorted.upsert([VectorRecord("a", (1.0,)), VectorRecord("b", (2.0,))])
    with pytest.raises(ComponentError, match="results for top_k"):
        unsorted.search((1.0,), top_k=1)
    with pytest.raises(ComponentError, match="sorted"):
        unsorted.search((1.0,), top_k=5)


def test_sparse_indexer_contract(chunks: tuple[KnowledgeChunk, ...]) -> None:
    sparse = MemorySparseIndexer(name="sp")
    sparse.index_chunks([])
    sparse.index_chunks(chunks)
    assert sparse.count() == len(chunks)
    top = sparse.search("Retrieval systems rewrite queries", top_k=2)
    assert len(top) == 2 and top[0].score >= top[1].score
    assert sparse.artifact("h").kind == "sparse" and sparse.artifact("h").dimension is None
    with pytest.raises(ComponentError, match="non-empty"):
        sparse.search("   ")
    overflow = MemorySparseIndexer({"overflow": True})
    overflow.index_chunks(chunks)
    with pytest.raises(ComponentError, match="results for top_k"):
        overflow.search("query", top_k=1)


def test_hybrid_indexer_composes_and_fuses(chunks: tuple[KnowledgeChunk, ...]) -> None:
    store, sparse = MemoryVectorStore(), MemorySparseIndexer()
    hybrid = SumHybrid(store, sparse, name="hy")
    records = [VectorRecord(c.id, (1.0, float(i))) for i, c in enumerate(chunks)]
    with hybrid:
        assert store.is_ready and sparse.is_ready
        hybrid.index(chunks, records)
        assert hybrid.count() == len(chunks) == sparse.count()
        fused = hybrid.fuse([ScoredId("a", 1.0)], [ScoredId("a", 0.5), ScoredId("b", 1.0)], top_k=2)
        assert fused == [ScoredId("a", 1.5), ScoredId("b", 1.0)]
        assert hybrid.artifact("h").kind == "hybrid"
        with pytest.raises(ComponentError, match="must match"):
            hybrid.index(chunks, records[:-1])
        with pytest.raises(ComponentError, match="top_k"):
            hybrid.fuse([], [], top_k=-1)
    assert not store.is_ready and not sparse.is_ready
