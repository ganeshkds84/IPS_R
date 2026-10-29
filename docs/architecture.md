# Architecture

Sprint 1 provides the *scaffolding* that the frozen research architecture plugs
into. Milestone 1 implements the **dataset layer** ([below](#dataset-layer-milestone-1)).
No retrieval, ranking, policy or generation logic is implemented yet.

## Layers

```
cli.py ──► experiment/runner ──► pipeline/builder ──► registries ──► component implementations
   │              │   └─ execute_preparation ──► data.prepare_dataset   (data/: Milestone 1)
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
├── models.py        Document, DocumentElement, DocumentMetadata, Chunk, Question, DatasetMetadata
├── issues.py        ValidationIssue, IssueCode, Severity, IssueCollector, ValidationReport
├── validation.py    RecordSchema + SchemaTracker (raw records), DatasetValidator (canonical records)
├── io.py            fault-tolerant JSON / JSONL / TSV readers (gzip aware), JSONL writer
├── text.py          tokenisation, sentence segmentation, encoding checks, stable hashing
├── language.py      stop-word language identification
├── base.py          AbstractDataset + DATASETS registry
├── adapters/        squad, hotpotqa, natural_questions, ms_marco, academic_pdf
├── pdf/             backend.py → layout.py → metadata.py, composed by parser.py
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
   - classifies each line as title, authors, abstract, heading, caption, reference or
     paragraph, using font size and weight, numbering, known heading names and
     caption patterns;
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
                doc = self._build(lambda: Document(id=..., text=...), record_id=..., location=location)
                if doc is not None:
                    yield doc

    def _iter_questions(self): ...
```

Then import it in `adapters/__init__.py`, add `configs/dataset/my_dataset.yaml` and
add tests (build fixtures in `tests/data_factories.py`).
