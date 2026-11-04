"""Knowledge processing pipeline (Milestone 2): validated dataset → index-ready artifacts.

Normalisation → structural analysis → chunk processing → enrichment → (embedding /
index interfaces) → deterministic serialization with hashing, manifests and reports.
No retrieval is implemented here.
"""

from ips_r.knowledge.chunking import ChunkIssue, ChunkValidation, KnowledgeChunker
from ips_r.knowledge.embedding import (
    EMBEDDING_MODELS,
    AbstractEmbeddingModel,
    EmbeddingBatch,
    EmbeddingCache,
    EmbeddingRequest,
    EmbeddingResult,
)
from ips_r.knowledge.enrichment import ENRICHMENT_FIELDS, MetadataEnricher
from ips_r.knowledge.hashing import (
    HashDiff,
    artifact_hash,
    canonical_json,
    content_hash,
    diff_document_hashes,
    file_hash,
    text_hash,
)
from ips_r.knowledge.indexing import (
    HYBRID_INDEXERS,
    SPARSE_INDEXERS,
    VECTOR_STORES,
    AbstractHybridIndexer,
    AbstractSparseIndexer,
    AbstractVectorStore,
    IndexArtifact,
    RetrievalArtifact,
    ScoredId,
    VectorRecord,
)
from ips_r.knowledge.models import (
    KNOWLEDGE_SCHEMA_VERSION,
    KNOWLEDGE_TYPES,
    ChunkKind,
    Citation,
    CrossReference,
    Equation,
    Figure,
    Footnote,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeKind,
    Paragraph,
    Reference,
    Section,
    SourceSpan,
    Table,
    from_record,
    to_record,
)
from ips_r.knowledge.normalization import TextNormalizer, TrackedText
from ips_r.knowledge.pipeline import (
    IndexComponents,
    KnowledgeBuildResult,
    KnowledgeProcessor,
    build_knowledge_artifacts,
)
from ips_r.knowledge.serialization import (
    ArtifactWriter,
    load_manifest,
    read_table,
    verify_artifacts,
    write_manifest,
)
from ips_r.knowledge.statistics import KnowledgeStatisticsAccumulator, ProcessingReport
from ips_r.knowledge.structure import StructuralAnalyzer, StructureResult, TextUnit

__all__ = [
    "EMBEDDING_MODELS",
    "ENRICHMENT_FIELDS",
    "HYBRID_INDEXERS",
    "KNOWLEDGE_SCHEMA_VERSION",
    "KNOWLEDGE_TYPES",
    "SPARSE_INDEXERS",
    "VECTOR_STORES",
    "AbstractEmbeddingModel",
    "AbstractHybridIndexer",
    "AbstractSparseIndexer",
    "AbstractVectorStore",
    "ArtifactWriter",
    "ChunkIssue",
    "ChunkKind",
    "ChunkValidation",
    "Citation",
    "CrossReference",
    "EmbeddingBatch",
    "EmbeddingCache",
    "EmbeddingRequest",
    "EmbeddingResult",
    "Equation",
    "Figure",
    "Footnote",
    "HashDiff",
    "IndexArtifact",
    "IndexComponents",
    "KnowledgeBuildResult",
    "KnowledgeChunk",
    "KnowledgeChunker",
    "KnowledgeDocument",
    "KnowledgeKind",
    "KnowledgeProcessor",
    "KnowledgeStatisticsAccumulator",
    "MetadataEnricher",
    "Paragraph",
    "ProcessingReport",
    "Reference",
    "RetrievalArtifact",
    "ScoredId",
    "Section",
    "SourceSpan",
    "StructuralAnalyzer",
    "StructureResult",
    "Table",
    "TextNormalizer",
    "TextUnit",
    "TrackedText",
    "VectorRecord",
    "artifact_hash",
    "build_knowledge_artifacts",
    "canonical_json",
    "content_hash",
    "diff_document_hashes",
    "file_hash",
    "from_record",
    "load_manifest",
    "read_table",
    "text_hash",
    "to_record",
    "verify_artifacts",
    "write_manifest",
]
