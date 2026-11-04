# Architecture

Sprint 1 provides the *scaffolding* that the frozen research architecture plugs
into. Milestone 1 implements the **dataset layer** ([below](#dataset-layer-milestone-1)).
Milestone 2 implements the **knowledge processing pipeline**
([below](#knowledge-processing-pipeline-milestone-2)). No embedding model, index,
retrieval, ranking, policy or generation logic is implemented yet.

## Layers

```
cli.py ──► experiment/runner ──► pipeline/builder ──► registries ──► component implementations
   │              │   ├─ execute_preparation ──► data.prepare_dataset              (data/: Milestone 1)
   │              │   └─ execute_knowledge ───► knowledge.build_knowledge_artifacts (knowledge/: Milestone 2)
   ▼              ▼
config/loader   core/{log, seed, paths, env}
```

- **core**: dependency-free infrastructure (errors, logging, env, paths, seeds, registry, data types).
- **config**: the `AppConfig` schema and Hydra compose-API loader.
- **components**: `AbstractComponent`, the shared base class.
- **interfaces**: one package per component kind, each with an abstract base and a registry.
- **pipeline**: `PipelineComponents`, `AbstractPipeline` and the config-driven builder.
- **experiment**: run lifecycle (directories, config snapshot, metadata, execution).

## Component interfaces

| Interface | Package | Public method(s) | Hook(s) to implement | Registry |
| --- | --- | --- | --- | --- |
| `AbstractDataset` | `ips_r.data` | `iter_documents()`, `iter_questions()`, `load_*()`, `describe()`, `queries()`, `documents()`, `references()` | `_iter_documents`, `_iter_questions` | `DATASETS` |
| `AbstractChunker` | `ips_r.data.chunking` | `chunk()`, `chunk_documents()` | `_split` | `CHUNKERS` |
| `AbstractRetriever` | `ips_r.retrieval` | `index()`, `retrieve()`, `retrieve_batch()` | `_index`, `_retrieve` | `RETRIEVERS` |
| `AbstractCandidateGenerator` | `ips_r.candidates` | `generate()` | `_generate` | `CANDIDATE_GENERATORS` |
| `AbstractDecisionPolicy` | `ips_r.policy` | `decide()` | `_decide` | `DECISION_POLICIES` |
| `AbstractReranker` | `ips_r.reranking` | `rerank()` | `_rerank` | `RERANKERS` |
| `AbstractGenerator` | `ips_r.generation` | `generate()`, `generate_batch()` | `_generate` | `GENERATORS` |
| `AbstractEvaluator` | `ips_r.evaluation` | `evaluate()` | `_evaluate` | `EVALUATORS` |
| `AbstractPipeline` | `ips_r.pipeline` | `run()` | `_run` | `PIPELINES` |
| `AbstractEmbeddingModel` | `ips_r.knowledge` | `embed()`, `embed_chunks()` | `dimension`, `_embed` | `EMBEDDING_MODELS` |
| `AbstractVectorStore` | `ips_r.knowledge` | `upsert()`, `delete()`, `count()`, `search()` | `_upsert`, `_delete`, `_count`, `_search` | `VECTOR_STORES` |
| `AbstractSparseIndexer` | `ips_r.knowledge` | `index_chunks()`, `count()`, `search()` | `_index`, `_count`, `_search` | `SPARSE_INDEXERS` |
| `AbstractHybridIndexer` | `ips_r.knowledge` | `index()`, `fuse()`, `count()` | `_fuse` | `HYBRID_INDEXERS` |

All of them follow the **template method** pattern. The public method:

1. validates arguments (e.g. `top_k > 0`, aligned predictions/references),
2. runs `setup()` lazily the first time,
3. logs start/end with timing,
4. calls the protected hook,
5. checks the output contract (e.g. at most `top_k` results, reranker output ⊆ input),
6. wraps unexpected exceptions in `ComponentError`, chaining the original as `__cause__`.

Implementations therefore contain only research logic.

### Data contracts

`ips_r.core.types` defines immutable containers: `Query`, `Document`,
`ScoredDocument`, `Candidate`, `Decision`, `Generation`, `Reference` and
`EvaluationResult`. They are **provisional**. Each carries a read-only
`metadata` mapping, so later sprints can attach extra information without
changing the interfaces.

### Lifecycle

```python
with MyRetriever({"k1": 1.2}) as retriever:   # setup() ... teardown()
    retriever.index(documents)
    hits = retriever.retrieve(query, top_k=10)
```

Override `_setup` / `_teardown` to load or release models and indices.
`AbstractPipeline` cascades both to all of its components.

## Adding an implementation (later sprints)

1. Implement the interface and register it:

   ```python
   # src/ips_r/retrieval/my_retriever.py
   from ips_r.retrieval import RETRIEVERS, AbstractRetriever

   @RETRIEVERS.register("my_retriever")
   class MyRetriever(AbstractRetriever):
       def _index(self, documents): ...
       def _retrieve(self, query, top_k): ...
   ```

2. Import the module from the package's `__init__.py` so that registration runs.
3. Add `configs/retriever/my_retriever.yaml`:

   ```yaml
   name: my_retriever
   params:
     k1: 1.2
   ```

4. Select it with `ips-r run retriever=my_retriever`.
5. Add tests under `tests/unit/`.

## Dataset layer (Milestone 1)

```
ips_r/data/
├── models.py        Document, DocumentMetadata, Question, DatasetMetadata (re-exports all models)
├── elements.py      ElementType, BBox, ElementDraft, DocumentElement
├── chunks.py        Chunk
├── fields.py        field validators shared by the models
├── issues.py        ValidationIssue, IssueCode, Severity, IssueCollector, ValidationReport
├── validation.py    RecordSchema + SchemaTracker (raw records), DatasetValidator (canonical records)
├── io.py            fault-tolerant JSON / JSONL / TSV readers (gzip aware), JSONL writer
├── text.py          tokenisation, sentence segmentation, encoding checks, stable hashing
├── language.py      stop-word language identification
├── base.py          AbstractDataset + DATASETS registry
├── adapters/        squad, hotpotqa, natural_questions, ms_marco, academic_pdf
├── pdf/             backend.py → layout.py (+ layout_settings.py, lines.py) → metadata.py,
│                    composed by parser.py
├── chunking/        AbstractChunker + 4 strategies + semantic interface, CHUNKERS registry
├── statistics.py    StatisticsAccumulator → DatasetStatistics (JSON + Markdown)
└── preparation.py   prepare_dataset(): the streaming pipeline that writes all artefacts
```

### Data flow

```
source files ─► adapter._iter_documents / _iter_questions ─► canonical models
                  │ (raw-record problems → ValidationIssue; record skipped)
                  ▼
prepare_dataset:  DatasetValidator ─► drop error records ─► chunker.chunk ─► StatisticsAccumulator
                  │                                                              │
                  ▼                                                              ▼
       validation_report.{json,md}     documents / questions / chunks .jsonl, statistics.{json,md}, metadata.json
```

Everything streams, so memory grows with ids and the vocabulary, not with corpus text.
`AbstractDataset.load_documents()` and `load_questions()` materialise a list when one
is needed, and cache it until `teardown()`.

### Canonical models

All models are frozen dataclasses. They check *structural* invariants in
`__post_init__` and raise `ValueError` (non-empty ids, element spans that match
the text, valid page ranges). They round-trip through `to_dict()` / `from_dict()`.

- **Document**: `id`, `text` and `metadata` (title, authors, year, venue, DOI,
  keywords, abstract, language, URL). Optionally also `elements`, `source`,
  `num_pages` and `extra`. Each `DocumentElement` (title, authors, abstract, heading,
  paragraph, caption, table, reference) occupies exactly `text[start:end]` and records
  its page, section heading, heading level and, for tables, its cells.
  `Document.from_elements()` assembles the text from elements.
- **Chunk**: exactly `document.text[start:end]`, with id `"<document_id>::<index:05d>"`.
  Page range and section are derived from the elements it overlaps.
- **Question**: answers, supporting document ids, a type and extras.
- **DatasetMetadata**: provenance, counts and a `sha256:` content hash of the
  accepted records.

`Document.to_core()`, `Chunk.to_core()`, `Question.to_query()` and
`Question.to_reference()` convert to the retrieval contracts in `ips_r.core.types`.
`AbstractDataset.queries()`, `documents()` and `references()` keep the Sprint 1 API
working.

### Validation

Validation happens at two levels, and both *report* issues instead of raising:

| Level | Where | Detects |
| --- | --- | --- |
| Raw records | adapters via `RecordSchema`, `SchemaTracker`, `io` readers | missing fields, malformed records, wrong types, invalid UTF-8, unparseable JSON/TSV lines, inconsistent schemas, corrupted PDFs |
| Canonical records | `DatasetValidator` | duplicate ids, empty text, encoding defects (U+FFFD, surrogates, control characters, mojibake), invalid metadata (year range, DOI, language code, URL, blank names), dangling supporting-document references |

Issues have a code, a severity (`error`/`warning`/`info`), a record id and a
location (`file:line` or JSON path). `ValidationReport` counts every issue and
stores up to `max_issues_per_code` examples. Records with errors are dropped from
the outputs when `preparation.drop_invalid` is set. With `strict: true`, a dataset
raises on its first ingestion error.

### PDF ingestion

`PDFParser` composes three stages, and each can be replaced:

1. **Backend** (`PDF_BACKENDS`; default `pdfplumber`). Extracts positioned words with
   font size and name, ruled tables, and the info dictionary into the backend-neutral
   `RawPDF`. Corrupted, encrypted or empty files raise `PDFParseError`.
2. **Layout analysis** (`LayoutAnalyzer`; pure and rule-based):
   - groups words into lines, leaving out words that fall inside tables;
   - orders two-column pages as header → left column → right column;
   - drops page-number lines;
   - classifies each line as title, authors, abstract, heading, caption, reference,
     equation, footnote or paragraph, using font size and weight, numbering, known
     heading names, caption patterns, mathematical-symbol density and page position;
   - merges lines into blocks using vertical gaps, repairing hyphenation;
   - assigns heading levels (numbering depth, otherwise font-size rank) and the
     enclosing section.

   Blocks never span pages, so every element has an exact page number.
3. **Metadata extraction** (`MetadataExtractor`):
   - title: typeset title or info dictionary, rejecting junk such as "Microsoft Word - …";
   - authors: info dictionary or the typeset author line;
   - year: copyright line → venue → arXiv id → creation date;
   - venue: "Proceedings of …", known acronyms or arXiv;
   - DOI, keywords and abstract;
   - language: stop-word profiles.

Scanned PDFs without a text layer are reported as `corrupted_pdf`, because OCR is
not part of this layer.

### Chunking

`AbstractChunker` subclasses implement `_split(text) -> spans` only. The base class
trims whitespace, drops chunks shorter than `min_chunk_chars`, rejects
out-of-range spans and builds `Chunk` objects. Strategies:

| Name | Unit | Overlap |
| --- | --- | --- |
| `recursive_character` | separator hierarchy (`\n\n`, `\n`, `. `, ` `, hard cut), merged up to `chunk_size` characters | trailing pieces ≤ `chunk_overlap` characters |
| `sentence` | `sentences_per_chunk` sentences (optional `max_chunk_chars`) | `sentence_overlap` sentences |
| `token` | `chunk_size` regex tokens | `chunk_overlap` tokens |
| `sliding_window` | `window_size` characters, snapped to word boundaries | `window_size − stride` characters |

`AbstractSemanticChunker` implements boundary placement over sentence similarities.
The scoring hook `_similarities` is left abstract because it needs embeddings, which
the dataset layer does not provide. The registered `semantic` chunker is a
placeholder that fails at construction with an explanation.

### Statistics

`DatasetStatistics` reports:
- counts of documents, pages, questions and chunks;
- length distributions (characters and tokens: mean, std, min, P50, P90, max) for
  documents, chunks, chunks per document, questions and answers per question;
- vocabulary statistics: total tokens, vocabulary size, type/token ratio, hapax
  legomena and top terms;
- exact-duplicate ratios for documents and chunks, after whitespace and case
  normalisation.

It is rendered as JSON and as Markdown.

### Adding a dataset adapter

```python
# src/ips_r/data/adapters/my_dataset.py
@DATASETS.register("my_dataset")
class MyDataset(AbstractDataset):
    description = "..."

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self._path = Path(self.str_param("path"))   # type-checked at construction

    def _setup(self):
        self._path = self._path_param("path")        # existence checked at setup

    def _iter_documents(self):
        for location, record in iter_jsonl(self._path, sink=self._record):
            if self._record_all(SCHEMA.check(record, location=location)):
                doc = self._build(partial(_document, record), record_id=..., location=location)
                if doc is not None:
                    yield doc

    def _iter_questions(self): ...
```

Then import it in `adapters/__init__.py`, add `configs/dataset/my_dataset.yaml` and
add tests (build fixtures in `tests/data_factories.py`).

## Knowledge processing pipeline (Milestone 2)

The knowledge pipeline turns a validated dataset into deterministic, index-ready
knowledge artifacts. It stops just before retrieval. Embedding and index components
exist only as interfaces.

```
ips_r/knowledge/
├── normalization.py   TextNormalizer + TrackedText (offset-preserving cleaning)
├── primitives.py      KnowledgeKind, ChunkKind, SourceSpan, shared validators
├── objects.py         Section, Paragraph, Figure, Table, Equation, Reference, Citation,
│                      CrossReference, Footnote
├── models.py          KnowledgeChunk, KnowledgeDocument, KNOWLEDGE_TYPES (re-exports the above)
├── records.py         to_record / from_record: flat JSON records of the objects
├── structure/         StructuralAnalyzer: analyzer.py (document walk), builders.py (one
│                      builder per object kind), mentions.py (citations, cross references),
│                      patterns.py, results.py (TextUnit, StructureResult)
├── chunking.py        KnowledgeChunker: section/table/caption-aware chunks, ids, links,
│                      duplicates, overlap validation
├── enrichment.py      MetadataEnricher: document and section context in chunk metadata
├── embedding.py       AbstractEmbeddingModel, EmbeddingRequest/Batch/Result, EmbeddingCache
├── indexing.py        AbstractVectorStore, AbstractSparseIndexer, AbstractHybridIndexer,
│                      IndexArtifact, RetrievalArtifact
├── hashing.py         canonical JSON, content / artifact hashes, HashDiff
├── processor.py       KnowledgeProcessor: one Document → one KnowledgeDocument
├── artifact_schema.py fixed table schemas (the only module that imports pyarrow)
├── serialization.py   ArtifactWriter (JSONL / Arrow / Parquet), manifest, checksums, verify
├── statistics.py      KnowledgeStatisticsAccumulator, ProcessingReport (JSON + Markdown)
└── pipeline.py        build_knowledge_artifacts(): the streaming orchestration
```

`ips_r.knowledge` builds on the dataset layer. It consumes `AbstractDataset`,
`DatasetValidator`, `AbstractChunker` and the `data.io` writers. It never imports
`cli`, `experiment` or `pipeline`.

### Data flow

```
dataset.iter_documents()
  └─► DatasetValidator ─► (drop invalid) ─► KnowledgeProcessor.process(document)
        TextNormalizer ─► StructuralAnalyzer ─► KnowledgeChunker ─► MetadataEnricher
  └─► KnowledgeStatisticsAccumulator
  └─► embedding / sparse index / vector store   (only if configured; none are registered)
  └─► ArtifactWriter ─► jsonl/ arrow/ parquet/ ─► manifest.json + checksums.sha256 + reports/
```

Documents stream one at a time. Arrow/Parquet writers buffer at most
`serialization.batch_size` records per table.

### Text normalisation

`TextNormalizer` applies configurable steps in a fixed order: Unicode normalisation,
control and zero-width character removal, ligature expansion, hyphen and dash
normalisation (soft hyphens, line-break de-hyphenation), quote normalisation,
optional punctuation removal, optional lower-casing and whitespace normalisation.
Every step runs on a `TrackedText`, which records for each output character the
span of original characters it came from. Cleaned text can therefore always be
mapped back to exact offsets in `Document.text`.

### Document structure

`StructuralAnalyzer` handles two kinds of document:

- *Structured* documents (PDFs): it walks the elements. Headings build the section
  tree, with level, numbering (`2.1`, `A.1`), parent, path and appendix flag.
  Paragraphs, abstracts, captions, tables, equations, references and footnotes become
  knowledge objects, and table captions are paired with their adjacent table.
- *Plain* documents (QA passages): they get a root section whose paragraphs are split
  on blank lines.

Citations (numeric `[1, 3-5]` and author-year `(Smith et al., 2020)`) and cross
references (`Figure 2`, `Table 1`, `Section 3.1`, `Eq. (4)`, `Appendix A`) are found in
running text. They are then resolved against the document's references, figures,
tables, sections and equations.

The PDF layout analysis (`ips_r.data.pdf`) also classifies display equations and
footnotes. An equation is a short line dense in mathematical symbols, or a numbered
`(n)` line containing an operator. A footnote is small text in the bottom zone of a
page, and a new footnote starts at a footnote marker. The resulting elements are
`ElementType.EQUATION` and `ElementType.FOOTNOTE`.

### Knowledge objects

Every object has a deterministic `id` (`<document_id>#<prefix><NNNN>`, numbered in
reading order), `document_id`, `order`, `section_id` (the parent section) and a
`SourceSpan`. The span holds the original-document `start`/`end`, the page range, the
bounding box and the index of the originating element. `KnowledgeDocument` groups
all objects of one document. Its `source_hash` covers the source content and its
`knowledge_hash` covers every derived object.

### Chunk processing

`KnowledgeChunker` cuts `TextUnit`s into `KnowledgeChunk`s:

| Feature | Behaviour |
| --- | --- |
| Section-aware | Running text is grouped per section, so chunks never cross a section boundary (`section_aware`) |
| Table-aware | A table and its caption form one chunk. Tables longer than `max_table_chars` are split by rows, repeating caption and header |
| Caption-aware | Figure captions are standalone chunks that are never split |
| Running text | Split by the Milestone 1 strategy selected with `chunker=...`, or kept whole with `chunker=none`. Equations stay in their section's running text |
| Deterministic ids | `<document_id>::<16 hex>`, hashed from the parent unit, unit span and chunk text |
| Links | `parent_id` / `section_id` (parent-child), `prev_id` / `next_id` (adjacent, per document) |
| Duplicates | Corpus-wide exact duplicates after whitespace and case folding: `mark` sets `duplicate_of`, `drop` removes them, `off` disables the check |
| Overlap validation | Consecutive pieces of a unit are reported as `contained`, `excessive_overlap` (overlap > `max_overlap_ratio` × the longer neighbour) or `gap` (uncovered non-whitespace text) |

`MetadataEnricher` then adds the configured fields to `KnowledgeChunk.metadata`:
title, authors, year, venue, DOI, language, section path and number, appendix flag,
pages, the citations and cross references inside the chunk, and the token count.

### Embedding and index interfaces

These are interfaces only. Nothing is registered in `EMBEDDING_MODELS`,
`VECTOR_STORES`, `SPARSE_INDEXERS` or `HYBRID_INDEXERS`.

| Interface | Template method | Hooks |
| --- | --- | --- |
| `AbstractEmbeddingModel` | `embed(batch)`: serves `EmbeddingCache` hits, embeds misses only, checks count, dimension and finiteness, optionally L2-normalises. `embed_chunks()` batches by `batch_size` | `dimension`, `_embed` |
| `AbstractVectorStore` | `upsert()` (unique ids, one dimension), `delete()`, `count()`, `search()` (contract only) | `_upsert`, `_delete`, `_count`, `_search` |
| `AbstractSparseIndexer` | `index_chunks()`, `count()`, `search()` (contract only) | `_index`, `_count`, `_search` |
| `AbstractHybridIndexer` | `index(chunks, records)` (ids must match), `fuse()` (contract only); wraps a vector store and a sparse indexer | `_fuse` |

`IndexArtifact` records the provenance of a built index. Its `source_artifact_hash`
ties the index to the knowledge artifact it was built from, so
`IndexArtifact.is_stale(hash)` detects that a document has changed since. Built
indexes are listed in `manifest.json`. `RetrievalArtifact` is the record format of a
ranked result list, unique and sorted by descending score.

### Serialization

| Path | Format |
| --- | --- |
| `jsonl/<table>.jsonl` | One JSON object per line, sorted keys |
| `arrow/<table>.arrow` | Apache Arrow IPC file |
| `parquet/<table>.parquet` | Parquet (`zstd` by default) |
| `manifest.json` | Schema version, artifact id and hash, package and pyarrow versions, settings fingerprints, dataset metadata, per-table record counts and files (bytes, SHA-256), per-document hashes, index artifacts |
| `checksums.sha256` | `<sha256>  <relative path>` for every data file and the manifest (`sha256sum -c` compatible) |
| `reports/` | `knowledge_statistics`, `processing_report`, `validation_report` (`.json` + `.md`). These contain timings and are not covered by the manifest |

Tables: `documents`, `questions`, `sections`, `paragraphs`, `figures`, `tables`,
`equations`, `references`, `citations`, `cross_references`, `footnotes`, `chunks`.
Every table is written even when it is empty. Schemas are fixed and derived from the
record dataclasses (`artifact_schema.py`). Nested values (spans, rows, metadata) are
stored as canonical JSON strings in Arrow and Parquet and decoded again by
`read_table()`. `verify_artifacts()` (CLI: `ips-r verify-artifacts`) re-hashes every
listed file.

### Hashing and invalidation

```
source document hash  +  processing fingerprint (normalisation, structure, chunking, enrichment, schema)
   → knowledge document hash → chunk content hash / chunk lineage hash
   → artifact hash (ordered document hashes + fingerprints + schema version)
   → IndexArtifact.source_artifact_hash
```

All hashes are SHA-256 over canonical JSON (sorted keys, compact, UTF-8). They do
not depend on `PYTHONHASHSEED`. `Document.source` (the file path) is excluded, so
moving the data directory does not invalidate artifacts. Changing a document changes
only that document's hash and the artifact hash. Changing a processing setting
changes every hash. `diff_document_hashes(old, new)` lists added, removed and
changed documents.

### Statistics

`KnowledgeStatisticsAccumulator` reports:

- document, page and per-document object counts;
- sections by level, and appendix sections;
- chunks by kind, character and token lengths, and duplicates;
- citations by style, resolution rate and most-cited references;
- cross references by kind and resolution rate;
- references with year, DOI and first author, and a year distribution;
- token totals, vocabulary size and type/token ratio.

`ProcessingReport` records document outcomes (seen, processed, dropped, failed),
per-stage status and timings, chunk-validation counts, characters in and out, and
any failures.
