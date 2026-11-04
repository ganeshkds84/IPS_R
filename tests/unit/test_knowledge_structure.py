"""Tests for knowledge models, structural analysis, chunk processing and enrichment."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from ips_r.config.schema import (
    ChunkProcessingConfig,
    EnrichmentConfig,
    ProcessingConfig,
    StructureConfig,
)
from ips_r.core.exceptions import ConfigurationError
from ips_r.data import CHUNKERS
from ips_r.data.models import Document, ElementDraft, ElementType
from ips_r.knowledge import (
    KNOWLEDGE_TYPES,
    ChunkKind,
    KnowledgeChunk,
    KnowledgeChunker,
    KnowledgeDocument,
    KnowledgeProcessor,
    MetadataEnricher,
    Section,
    SourceSpan,
    StructuralAnalyzer,
    TextNormalizer,
    from_record,
    to_record,
)
from ips_r.knowledge.structure import first_author
from tests.knowledge_factories import big_table_document, paper


@pytest.fixture(scope="module")
def knowledge() -> KnowledgeDocument:
    return KnowledgeProcessor().process(paper())


def _processor(**chunking: Any) -> KnowledgeProcessor:
    return KnowledgeProcessor(
        processing=ProcessingConfig(chunking=ChunkProcessingConfig(**chunking))
    )


# --------------------------------------------------------------------------- #
# Section hierarchy and objects (Part B / C)
# --------------------------------------------------------------------------- #
def test_section_hierarchy(knowledge: KnowledgeDocument) -> None:
    sections = {s.title: s for s in knowledge.sections[1:]}
    root = knowledge.sections[0]
    assert (
        root.level == 0 and root.parent_id is None and root.title == "A Synthetic Study of Intent"
    )
    details = sections["1.1 Details 1"]
    assert (details.number, details.level) == ("1.1", 2)
    assert details.parent_id == sections["1 Section 1"].id
    assert details.path == ("1 Section 1", "1.1 Details 1")
    appendix = sections["Appendix A Extra Results"]
    sub = sections["A.1 More Tables"]
    assert appendix.is_appendix and sub.is_appendix
    assert sub.parent_id == appendix.id and sub.number == "A.1"
    assert not sections["References"].is_appendix
    assert all(s.source is not None and s.source.page for s in knowledge.sections[1:])


def test_objects_are_traceable_to_the_original_text(knowledge: KnowledgeDocument) -> None:
    original = paper()
    for paragraph in knowledge.paragraphs:
        assert paragraph.source is not None
        raw = original.text[paragraph.source.start : paragraph.source.end]
        assert TextNormalizer().normalize_text(raw) == paragraph.text
        element = original.elements[paragraph.source.element_index or 0]
        assert element.start <= paragraph.source.start and paragraph.source.end <= element.end
    orders = [p.order for p in knowledge.paragraphs]
    assert orders == sorted(orders)
    assert knowledge.paragraphs[2].text == 'We study the efficient "preservation" of intent.'
    assert [p.role for p in knowledge.paragraphs[:3]] == ["title", "authors", "abstract"]


def test_figures_tables_equations_references_footnotes(knowledge: KnowledgeDocument) -> None:
    (figure,) = knowledge.figures
    assert (figure.label, figure.number, figure.source and figure.source.page) == (
        "Figure 1",
        "1",
        2,
    )
    (table,) = knowledge.tables
    assert table.caption == "Table 1: Main results." and table.number == "1"
    assert table.rows == (("Model", "Score"), ("BM25", "0.41"))
    assert table.caption_source is not None
    assert [e.number for e in knowledge.equations] == ["1", "2"]
    first, second = knowledge.references
    assert (first.marker, first.first_author, first.year, first.doi) == ("1", "Doe", 2019, None)
    assert (second.first_author, second.year, second.doi) == ("Smith", 2020, "10.1000/xyz.1")
    (footnote,) = knowledge.footnotes
    assert footnote.marker == "1"


def test_citations_and_cross_references_resolve(knowledge: KnowledgeDocument) -> None:
    numeric = [c for c in knowledge.citations if c.style == "numeric"]
    author_year = [c for c in knowledge.citations if c.style == "author_year"]
    assert numeric and all(c.target_ids == (knowledge.references[0].id,) for c in numeric)
    assert author_year and all(c.target_ids == (knowledge.references[1].id,) for c in author_year)
    assert all(c.resolved for c in knowledge.citations)
    kinds = {(x.target_kind, x.target_number): x.target_id for x in knowledge.cross_references}
    assert kinds[("figure", "1")] == knowledge.figures[0].id
    assert kinds[("table", "1")] == knowledge.tables[0].id
    assert kinds[("equation", "1")] == knowledge.equations[0].id
    assert kinds[("appendix", "A")] is not None
    assert kinds[("section", "2.1")] == next(s.id for s in knowledge.sections if s.number == "2.1")
    original = paper()
    for citation in knowledge.citations:
        assert citation.source is not None
        assert original.text[citation.source.start : citation.source.end] == citation.marker


def test_citation_parsing_edge_cases() -> None:
    doc = Document.from_elements(
        "d",
        [
            ElementDraft(
                ElementType.PARAGRAPH, "See [2-4], [7; 9] and [1-100] and (Nobody, 1999)."
            ),
            ElementDraft(ElementType.REFERENCE, "3. Smith J, Doe K. Title. 2001."),
        ],
    )
    result = StructuralAnalyzer(TextNormalizer()).analyze(doc)
    keys = {c.marker: c.keys for c in result.citations}
    assert keys["[2-4]"] == ("2", "3", "4")
    assert keys["[7; 9]"] == ("7", "9")
    assert keys["[1-100]"] == ()
    assert keys["(Nobody, 1999)"] == ("Nobody 1999",)
    by_marker = {c.marker: c.target_ids for c in result.citations}
    assert by_marker["[2-4]"] == (result.references[0].id,)
    assert result.references[0].marker == "3" and result.references[0].first_author == "Smith"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("B. Author. Title. 2021.", "Author"),
        ("Smith, J. A book.", "Smith"),
        ("J. Smith and K. Doe. Title.", "Smith"),
        ("vaswani a. lowercase", None),
    ],
)
def test_first_author(body: str, expected: str | None) -> None:
    assert first_author(body) == expected


def test_plain_documents_get_root_paragraphs() -> None:
    doc = Document(id="qa", text="First passage [1].\n\nSecond passage.\n \n")
    result = StructuralAnalyzer(TextNormalizer()).analyze(doc)
    assert [p.text for p in result.paragraphs] == ["First passage [1].", "Second passage."]
    assert len(result.sections) == 1
    (unit,) = result.units
    assert unit.kind is ChunkKind.TEXT
    assert result.citations[0].target_ids == ()


def test_detection_can_be_disabled() -> None:
    analyzer = StructuralAnalyzer(
        TextNormalizer(), StructureConfig(detect_citations=False, detect_cross_references=False)
    )
    result = analyzer.analyze(paper())
    assert result.citations == [] and result.cross_references == []
    with pytest.raises(ConfigurationError, match="pattern"):
        StructuralAnalyzer(TextNormalizer(), StructureConfig(numeric_citation_pattern="("))


def test_unpaired_table_caption_and_unlabelled_figure() -> None:
    doc = Document.from_elements(
        "d",
        [
            ElementDraft(ElementType.CAPTION, "Table 3: Caption without a ruled table.", page=1),
            ElementDraft(ElementType.CAPTION, "Overview of results.", page=1),
            ElementDraft(ElementType.HEADING, "A Study of Things", page=1, level=1),
        ],
        num_pages=1,
    )
    result = StructuralAnalyzer(TextNormalizer()).analyze(doc)
    assert result.tables[0].number == "3" and result.tables[0].rows == ()
    assert result.figures[0].label is None
    assert result.sections[1].number is None  # "A Study" is not letter-numbered outside appendices


# --------------------------------------------------------------------------- #
# Chunk processing (Part D)
# --------------------------------------------------------------------------- #
def test_chunks_are_linked_traceable_and_deterministic(knowledge: KnowledgeDocument) -> None:
    chunks = knowledge.chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].prev_id is None and chunks[-1].next_id is None
    for a, b in itertools.pairwise(chunks):
        assert a.next_id == b.id and b.prev_id == a.id
    ids = {s.id for s in knowledge.sections}
    assert all(c.section_id in ids for c in chunks)
    kinds = {c.kind for c in chunks}
    assert {
        ChunkKind.FRONT_MATTER,
        ChunkKind.TEXT,
        ChunkKind.FIGURE,
        ChunkKind.TABLE,
        ChunkKind.FOOTNOTE,
    } <= kinds
    assert ChunkKind.REFERENCE not in kinds
    again = KnowledgeProcessor().process(paper())
    assert [c.id for c in again.chunks] == [c.id for c in chunks]
    assert again.knowledge_hash == knowledge.knowledge_hash
    original = paper()
    for chunk in chunks:
        if chunk.kind is ChunkKind.TEXT:
            raw = original.text[chunk.source.start : chunk.source.end]
            assert TextNormalizer().normalize_text(raw).startswith(chunk.text[:20])
        assert chunk.lineage_hash and chunk.content_hash


def test_duplicate_detection_modes() -> None:
    marked = KnowledgeProcessor().process(paper())
    duplicates = [c for c in marked.chunks if c.duplicate_of is not None]
    assert len(duplicates) == 1  # section 2 repeats section 1's body text
    dropped = _processor(deduplicate="drop").process(paper())
    assert len(dropped.chunks) == len(marked.chunks) - 1
    off = _processor(deduplicate="off").process(paper())
    assert all(c.duplicate_of is None for c in off.chunks)


def test_strategy_based_chunking_never_crosses_sections() -> None:
    processor = KnowledgeProcessor(
        base_chunker=CHUNKERS.create("token", {"chunk_size": 12, "chunk_overlap": 4})
    )
    knowledge = processor.process(paper(body_repeats=4))
    text_chunks = [c for c in knowledge.chunks if c.kind is ChunkKind.TEXT]
    assert len(text_chunks) > 8
    by_unit: dict[str, list[KnowledgeChunk]] = {}
    for chunk in text_chunks:
        by_unit.setdefault(chunk.parent_id, []).append(chunk)
        assert chunk.parent_id == chunk.section_id
    assert any(len(group) > 1 for group in by_unit.values())
    assert processor.chunker.validation.counts["excessive_overlap"] == 0


def test_table_aware_splitting_repeats_caption_and_header() -> None:
    processor = _processor(max_table_chars=300)
    chunks = [
        c for c in processor.process(big_table_document()).chunks if c.kind is ChunkKind.TABLE
    ]
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text.startswith("Table 1: All values.\nid | value\n")
        assert len(chunk.text) <= 300 + 40
    assert len({c.unit_start for c in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("settings", "absent"),
    [
        ({"caption_aware": False}, ChunkKind.FIGURE),
        ({"table_aware": False}, ChunkKind.TABLE),
        ({"include_footnotes": False}, ChunkKind.FOOTNOTE),
        ({"include_front_matter": False}, ChunkKind.FRONT_MATTER),
    ],
)
def test_awareness_switches(settings: dict[str, Any], absent: ChunkKind) -> None:
    kinds = {c.kind for c in _processor(**settings).process(paper()).chunks}
    assert absent not in kinds


def test_references_and_section_unaware_mode() -> None:
    with_refs = _processor(include_references=True).process(paper())
    assert sum(c.kind is ChunkKind.REFERENCE for c in with_refs.chunks) == 2
    flat = _processor(section_aware=False).process(paper())
    root = flat.sections[0].id
    assert {c.parent_id for c in flat.chunks if c.kind is ChunkKind.TEXT} == {root}


def test_overlap_validation_flags_problems() -> None:
    sliding = CHUNKERS.create("sliding_window", {"window_size": 40, "stride": 10})
    processor = KnowledgeProcessor(
        processing=ProcessingConfig(chunking=ChunkProcessingConfig(max_overlap_ratio=0.2)),
        base_chunker=sliding,
    )
    processor.process(paper())
    assert processor.chunker.validation.counts["excessive_overlap"] > 0
    report = processor.chunker.validation.to_dict()
    assert report["examples"]
    processor.reset()
    assert not processor.chunker.validation.counts


@pytest.mark.parametrize(
    "settings",
    [{"deduplicate": "maybe"}, {"max_table_chars": 0}, {"max_overlap_ratio": 1.5}],
)
def test_invalid_chunk_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        KnowledgeChunker(ChunkProcessingConfig(**settings))


def test_fingerprints_change_with_settings() -> None:
    base = KnowledgeProcessor()
    assert base.fingerprint == KnowledgeProcessor().fingerprint
    assert base.fingerprint != _processor(max_table_chars=10).fingerprint
    chunked = KnowledgeProcessor(base_chunker=CHUNKERS.create("token", {}))
    assert base.fingerprint != chunked.fingerprint
    assert set(base.fingerprints) >= {"normalization", "structure", "chunking", "enrichment"}


# --------------------------------------------------------------------------- #
# Enrichment and models
# --------------------------------------------------------------------------- #
def test_enrichment_fields(knowledge: KnowledgeDocument) -> None:
    chunk = next(c for c in knowledge.chunks if c.metadata["citations"])
    meta = chunk.metadata
    assert meta["title"] == "A Synthetic Study of Intent"
    assert meta["authors"] == ["Grace Hopper"]
    assert meta["section_path"] == ["1 Section 1"]
    assert meta["section_number"] == "1"
    assert meta["pages"] == [1, 1]
    assert meta["token_count"] > 10
    assert {"marker": "[1]", "targets": [knowledge.references[0].id]} in meta["citations"]
    assert any(x["kind"] == "figure" for x in meta["cross_references"])
    appendix_chunk = next(c for c in knowledge.chunks if c.metadata["is_appendix"])
    assert appendix_chunk.metadata["section_number"] == "A.1"


def test_enrichment_field_selection() -> None:
    processor = KnowledgeProcessor(
        processing=ProcessingConfig(enrichment=EnrichmentConfig(fields=["title"]))
    )
    assert set(processor.process(paper()).chunks[0].metadata) == {"title"}
    with pytest.raises(ConfigurationError, match="Unknown"):
        MetadataEnricher(EnrichmentConfig(fields=["colour"]))
    with pytest.raises(ConfigurationError, match="token_pattern"):
        MetadataEnricher(EnrichmentConfig(token_pattern="("))


def test_record_round_trip_for_every_object(knowledge: KnowledgeDocument) -> None:
    for name, items in knowledge.objects().items():
        for item in items:
            assert from_record(KNOWLEDGE_TYPES[name], to_record(item)) == item
    summary = knowledge.summary_record()
    assert summary["num_chunks"] == len(knowledge.chunks)


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        SourceSpan("d", 5, 2)
    with pytest.raises(ValueError):
        SourceSpan("d", 0, 1, page=2, page_end=1)
    with pytest.raises(ValueError, match="root"):
        Section(id="s", document_id="d", order=0, level=1)
    with pytest.raises(ValueError, match="root"):
        KnowledgeDocument(
            id="d", source_hash="h", knowledge_hash="k", metadata=paper().metadata, sections=()
        )
    span = SourceSpan("d", 0, 1)
    chunk = KnowledgeChunk(
        id="c",
        document_id="d",
        index=0,
        text="x",
        kind=ChunkKind.TEXT,
        section_id="s",
        parent_id="s",
        unit_start=0,
        unit_end=1,
        source=span,
    )
    with pytest.raises(ValueError, match="duplicate"):
        KnowledgeDocument(
            id="d",
            source_hash="h",
            knowledge_hash="k",
            metadata=paper().metadata,
            sections=(Section(id="s", document_id="d", order=0, level=0),),
            chunks=(chunk, chunk),
        )
    with pytest.raises(ValueError):
        KnowledgeChunk(
            id="c",
            document_id="d",
            index=0,
            text="x",
            kind=ChunkKind.TEXT,
            section_id="s",
            parent_id="s",
            unit_start=1,
            unit_end=1,
            source=span,
        )
