"""Structured-config schema validated by OmegaConf at composition time.

The YAML files under ``configs/`` are merged on top of :class:`AppConfig`, so
typos in keys and type mismatches fail fast. Component-specific options live
in the free-form ``params`` mapping of each :class:`ComponentConfig`.
"""

# NOTE: no ``from __future__ import annotations`` - OmegaConf inspects real annotations.

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from ips_r.core.log import get_logger

logger = get_logger(__name__)

COMPONENT_KEYS: tuple[str, ...] = (
    "dataset",
    "chunker",
    "retriever",
    "candidate_generator",
    "decision_policy",
    "reranker",
    "generator",
    "evaluator",
)
"""Config keys of the pluggable components, in pipeline order."""


@dataclass
class ExperimentConfig:
    """Experiment identity, used for output directories and run metadata."""

    name: str = "default"
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PathsConfig:
    """Filesystem locations. Relative paths are resolved against ``root``."""

    root: str = MISSING
    data: str = MISSING
    cache: str = MISSING
    models: str = MISSING
    outputs: str = MISSING
    experiments: str = MISSING


@dataclass
class LoggingConfig:
    """Logging behaviour for a run."""

    level: str = "INFO"
    format: str = "console"
    to_file: bool = True


@dataclass
class SeedConfig:
    """Global random seed settings."""

    value: int = 42
    deterministic: bool = False


@dataclass
class RunConfig:
    """Per-invocation run settings."""

    output_dir: str | None = None
    dry_run: bool = False


@dataclass
class ComponentConfig:
    """Selection of a registered component implementation.

    Attributes:
        name: Registry name of the implementation, or ``None`` to disable the component.
        params: Keyword configuration passed to the implementation's constructor.
    """

    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Rules applied by :class:`ips_r.data.validation.DatasetValidator`."""

    check_encoding: bool = True
    check_metadata: bool = True
    min_year: int = 1500
    max_year: int = 2100
    max_issues_per_code: int = 100


@dataclass
class StatisticsConfig:
    """Options of :class:`ips_r.data.statistics.StatisticsAccumulator`."""

    word_pattern: str = r"\w+"
    lowercase: bool = True
    top_k_terms: int = 25


@dataclass
class PreparationConfig:
    """Behaviour of :func:`ips_r.data.preparation.prepare_dataset` (``ips-r prepare``).

    Attributes:
        output_subdir: Directory inside the run directory receiving the artefacts.
        drop_invalid: Exclude records with error-severity issues from the outputs.
        fail_on_error: Fail the run (after writing reports) if validation finds errors.
        write_documents: Write ``documents.jsonl``.
        write_questions: Write ``questions.jsonl``.
        write_chunks: Write ``chunks.jsonl`` (requires a chunker).
        validation: Validation rules.
        statistics: Statistics options.
    """

    output_subdir: str = "dataset"
    drop_invalid: bool = True
    fail_on_error: bool = False
    write_documents: bool = True
    write_questions: bool = True
    write_chunks: bool = True
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)


@dataclass
class NormalizationConfig:
    """Text cleaning and normalisation (:class:`ips_r.knowledge.normalization.TextNormalizer`).

    Steps run in a fixed order: Unicode form → control characters → ligatures →
    hyphens → quotes → punctuation → lowercase → whitespace.
    """

    unicode_form: str = "NFC"
    remove_control_chars: bool = True
    remove_zero_width: bool = True
    expand_ligatures: bool = True
    ligatures: dict[str, str] = field(default_factory=dict)
    normalize_hyphens: bool = True
    dehyphenate: bool = True
    dash_replacement: str = "-"
    normalize_quotes: bool = True
    normalize_whitespace: bool = True
    preserve_newlines: bool = True
    lowercase: bool = False
    preserve_punctuation: bool = True


@dataclass
class StructureConfig:
    """Structural analysis (:class:`ips_r.knowledge.structure.StructuralAnalyzer`)."""

    detect_citations: bool = True
    detect_cross_references: bool = True
    numeric_citation_pattern: str = r"\[(\d+(?:\s*[-\u2013,;]\s*\d+)*)\]"
    author_year_citation_pattern: str = (
        r"\(((?:[A-Z][A-Za-z'\-]+(?: et al\.)?(?: (?:and|&) [A-Z][A-Za-z'\-]+)?,? "
        r"(?:19|20)\d{2}[a-z]?(?:;\s*)?)+)\)"
    )
    cross_reference_pattern: str = (
        r"\b(Figures?|Figs?\.|Tables?|Tab\.|Sections?|Sec\.|Equations?|Eqs?\.|Appendix)"
        r"\s*\(?([A-Z]?\d+(?:\.\d+)*|[A-Z](?![a-z]))\)?"
    )
    section_number_pattern: str = r"^((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+(?=\S)"
    appendix_heading_pattern: str = r"^(?:appendix|appendices|supplementary material)\b"
    paragraph_separator_pattern: str = r"\n\s*\n"


@dataclass
class ChunkProcessingConfig:
    """Section/table/caption-aware chunk processing (:mod:`ips_r.knowledge.chunking`)."""

    section_aware: bool = True
    table_aware: bool = True
    caption_aware: bool = True
    max_table_chars: int = 2000
    include_front_matter: bool = True
    include_references: bool = False
    include_footnotes: bool = True
    max_overlap_ratio: float = 0.5
    deduplicate: str = "mark"


@dataclass
class EnrichmentConfig:
    """Chunk metadata enrichment (:class:`ips_r.knowledge.enrichment.MetadataEnricher`)."""

    fields: list[str] = field(
        default_factory=lambda: [
            "title",
            "authors",
            "year",
            "venue",
            "doi",
            "language",
            "section_path",
            "section_number",
            "is_appendix",
            "pages",
            "citations",
            "cross_references",
            "token_count",
        ]
    )
    token_pattern: str = r"\w+|[^\w\s]"


@dataclass
class ProcessingConfig:
    """Knowledge processing: structure analysis, chunk processing and enrichment."""

    structure: StructureConfig = field(default_factory=StructureConfig)
    chunking: ChunkProcessingConfig = field(default_factory=ChunkProcessingConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)


@dataclass
class SerializationConfig:
    """Artifact export (:mod:`ips_r.knowledge.serialization`)."""

    formats: list[str] = field(default_factory=lambda: ["jsonl", "parquet"])
    batch_size: int = 1000
    parquet_compression: str = "zstd"
    write_checksums: bool = True


@dataclass
class IndexConfig:
    """Index component selections (interfaces only in Milestone 2)."""

    vector_store: ComponentConfig = field(default_factory=ComponentConfig)
    sparse: ComponentConfig = field(default_factory=ComponentConfig)
    hybrid: ComponentConfig = field(default_factory=ComponentConfig)


@dataclass
class KnowledgeConfig:
    """Behaviour of :func:`ips_r.knowledge.pipeline.build_knowledge_artifacts`."""

    output_subdir: str = "knowledge"
    drop_invalid: bool = True
    fail_on_error: bool = False
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)


@dataclass
class AppConfig:
    """Root configuration schema."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    run: RunConfig = field(default_factory=RunConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    embedding: ComponentConfig = field(default_factory=ComponentConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    dataset: ComponentConfig = field(default_factory=ComponentConfig)
    chunker: ComponentConfig = field(default_factory=ComponentConfig)
    retriever: ComponentConfig = field(default_factory=ComponentConfig)
    candidate_generator: ComponentConfig = field(default_factory=ComponentConfig)
    decision_policy: ComponentConfig = field(default_factory=ComponentConfig)
    reranker: ComponentConfig = field(default_factory=ComponentConfig)
    generator: ComponentConfig = field(default_factory=ComponentConfig)
    evaluator: ComponentConfig = field(default_factory=ComponentConfig)
    pipeline: ComponentConfig = field(default_factory=ComponentConfig)
