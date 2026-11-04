# Configuration

Configuration is composed by [Hydra](https://hydra.cc) from `configs/` and
validated against the structured schema `ips_r.config.schema.AppConfig`.
Unknown keys and type mismatches are rejected when the config is composed.

## Structure

```
configs/
├── config.yaml              primary config + defaults list
├── paths/default.yaml       filesystem layout
├── logging/default.yaml     level, format, file logging
├── seed/default.yaml        global seed, determinism
├── preparation/default.yaml dataset preparation, validation and statistics settings
├── normalization/default.yaml   text cleaning and normalisation (knowledge pipeline)
├── processing/default.yaml      structure detection, chunk processing, enrichment
├── serialization/default.yaml   artifact formats, batch size, compression, checksums
├── knowledge/default.yaml       `ips-r build-knowledge` behaviour, validation, statistics
├── embedding/none.yaml          embedding model (interface only; none registered)
├── index/none.yaml              vector_store / sparse / hybrid (interfaces only; none registered)
├── experiment/              optional presets (# @package _global_)
│   └── example.yaml
├── dataset/                 ┐ none | squad | hotpotqa | natural_questions | ms_marco | academic_pdf
├── chunker/                 │ none | recursive_character | sentence | token | sliding_window | semantic
├── retriever/none.yaml      │
├── candidate_generator/…    │  one group per component kind;
├── decision_policy/…        │  `none` = disabled (name: null)
├── reranker/…               │
├── generator/…              │
├── evaluator/…              │
└── pipeline/none.yaml       ┘
```

Every component node has the same shape:

```yaml
name: <registry name or null>
params: {}   # free-form, passed to the implementation's constructor
```

Implementations check their `params` when they are constructed: types, ranges,
unknown PDF-parser keys and regex syntax. A misconfiguration therefore fails at
`ips-r run --dry-run` or at the start of `ips-r prepare`, not halfway through a dataset.

## Dataset layer

### Datasets (`configs/dataset/`)

Every dataset accepts these common parameters:

| Param | Default | Effect |
| --- | --- | --- |
| `split`, `version` | per file | Recorded in `metadata.json` |
| `max_documents`, `max_questions` | `null` | Stop after N records (quick experiments) |
| `strict` | `false` | Raise on the first error-severity ingestion issue instead of reporting it |

The adapter-specific parameters below point at files under `${paths.data}/raw/...`
by default:

| Dataset | Params |
| --- | --- |
| `squad` | `path` |
| `hotpotqa` | `path` |
| `natural_questions` | `path` (`.jsonl` / `.jsonl.gz`), `strip_html` |
| `ms_marco` | `collection_path`, `queries_path`, `qrels_path`, `only_judged_queries` |
| `academic_pdf` | `directory`, `glob`, `id_strategy` (`stem` / `relative_path`), `questions_path`, `parser.{backend, backend_options, layout, metadata}` |

`parser.layout` and `parser.metadata` accept any field of
`ips_r.data.pdf.LayoutSettings` and `MetadataSettings` respectively, for example
`heading_size_ratio`, `paragraph_gap_ratio`, `detect_columns`, `venue_acronyms` or
`use_creation_date_year`. Unknown keys are rejected.

### Chunkers (`configs/chunker/`)

| Chunker | Params |
| --- | --- |
| `recursive_character` | `chunk_size`, `chunk_overlap`, `separators` |
| `sentence` | `sentences_per_chunk`, `sentence_overlap`, `max_chunk_chars`, `abbreviations` |
| `token` | `chunk_size`, `chunk_overlap`, `token_pattern` |
| `sliding_window` | `window_size`, `stride`, `respect_word_boundaries` |
| `semantic` | interface placeholder: selecting it fails with an explanatory error |

Every chunker also accepts `min_chunk_chars`.

### Preparation (`configs/preparation/default.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `output_subdir` | `dataset` | Artefacts go to `<run_dir>/<output_subdir>/` |
| `drop_invalid` | `true` | Leave records with error-severity issues out of the outputs |
| `fail_on_error` | `false` | Exit non-zero (after writing reports) if validation finds errors |
| `write_documents` / `write_questions` / `write_chunks` | `true` | Toggle the JSONL outputs |
| `validation.check_encoding` / `check_metadata` | `true` | Enable the encoding and metadata checks |
| `validation.min_year` / `max_year` | `1500` / `2100` | Plausible publication years |
| `validation.max_issues_per_code` | `100` | Example issues stored per code (all issues are counted) |
| `statistics.word_pattern` / `lowercase` / `top_k_terms` | `\w+` / `true` / `25` | Vocabulary statistics |

```bash
uv run ips-r prepare dataset=ms_marco dataset.params.max_documents=100000 \
    chunker=token chunker.params.chunk_size=128 preparation.fail_on_error=true
```

## Knowledge pipeline

These groups configure `ips-r build-knowledge`. The running-text chunking strategy is
the Milestone 1 `chunker=` group; `chunker=none` keeps one chunk per section, table or
caption unit.

### Normalisation (`configs/normalization/default.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `unicode_form` | `NFC` | `NFC` / `NFKC` / `NFD` / `NFKD` / `none` |
| `remove_control_chars` / `remove_zero_width` | `true` / `true` | Drop C0/C1 controls (keeps newlines and tabs) and zero-width / bidi characters |
| `expand_ligatures`, `ligatures` | `true`, `{}` | Expand `ﬁ`, `ﬂ`, …; extra or overriding mappings |
| `normalize_hyphens`, `dehyphenate`, `dash_replacement` | `true`, `true`, `-` | Drop soft hyphens, join words broken at line ends (`repre-` + `sentation`), unify dash variants |
| `normalize_quotes` | `true` | Curly / angle quotes → ASCII |
| `normalize_whitespace`, `preserve_newlines` | `true`, `true` | Collapse whitespace runs; keep line and paragraph breaks |
| `lowercase` | `false` | Lower-case all text |
| `preserve_punctuation` | `true` | `false` replaces every non-word, non-space character with a space |

### Processing (`configs/processing/default.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `structure.detect_citations` / `detect_cross_references` | `true` | Detect and resolve in-text citations / cross references |
| `structure.*_pattern` | see file | Citation, cross-reference, section-number, appendix-heading and paragraph-separator regexes |
| `chunking.section_aware` / `table_aware` / `caption_aware` | `true` | See [architecture](architecture.md#chunk-processing) |
| `chunking.max_table_chars` | `2000` | Longer tables are split row-wise |
| `chunking.include_front_matter` / `include_references` / `include_footnotes` | `true` / `false` / `true` | Which content becomes chunks |
| `chunking.max_overlap_ratio` | `0.5` | Overlap validation threshold (× the longer neighbouring chunk) |
| `chunking.deduplicate` | `mark` | `mark` / `drop` / `off` corpus-wide exact duplicates |
| `enrichment.fields` | all 13 fields | Metadata fields added to every chunk |
| `enrichment.token_pattern` | `\w+\|[^\w\s]` | Tokens counted for `token_count` |

### Serialization (`configs/serialization/default.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `formats` | `[jsonl, parquet]` | Any of `jsonl`, `arrow`, `parquet` |
| `batch_size` | `1000` | Records per Arrow / Parquet record batch |
| `parquet_compression` | `zstd` | `zstd` / `snappy` / `gzip` / `none` |
| `write_checksums` | `true` | Write `checksums.sha256` |

### Knowledge run (`configs/knowledge/default.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `output_subdir` | `knowledge` | Artifacts go to `<run_dir>/<output_subdir>/` |
| `drop_invalid` | `true` | Skip documents and questions with error-severity validation issues |
| `fail_on_error` | `false` | Exit non-zero (after writing everything) on validation errors or failed documents |
| `validation.*`, `statistics.*` | as in `preparation` | Validation checks and token statistics |

### Embedding and index (`configs/embedding/`, `configs/index/`)

`embedding` is a component node (`name` + `params`). `index` holds three nodes:
`vector_store`, `sparse` and `hybrid`. A hybrid indexer requires both a vector store
and a sparse indexer. No implementation is registered in Milestone 2, so only
`none.yaml` exists. Selecting a name fails at the start of `ips-r build-knowledge`,
with the list of available implementations.

```bash
uv run ips-r build-knowledge dataset=academic_pdf chunker=sentence \
    processing.chunking.deduplicate=drop 'serialization.formats=[jsonl,arrow,parquet]'
```

## Loading

```bash
uv run ips-r config                                   # print resolved config
uv run ips-r config experiment=example seed.value=3   # with overrides
```

```python
from ips_r.config import load_config, as_app_config

cfg = load_config(["seed.value=3"])     # DictConfig, fully resolved
typed = as_app_config(cfg)              # AppConfig dataclass instance
```

The loader uses Hydra's compose API, so Hydra does **not** change the working
directory or take over logging. Run directories are managed by `ips_r.experiment`.

## Override cheat-sheet

| Goal | Override |
| --- | --- |
| Change a value | `seed.value=7` |
| Select a group option | `retriever=my_retriever`, `dataset=squad`, `chunker=sentence` |
| Change a knowledge setting | `normalization.lowercase=true`, `processing.chunking.max_table_chars=4000` |
| Change a component param | `chunker.params.chunk_size=512` |
| Apply a preset | `experiment=example` |
| Add a new component param | `+retriever.params.top_k=50` |
| Write the run to a fixed directory | `run.output_dir=/tmp/run1` |

## Environment variables

The `.env` file (see `.env.example`) is loaded at CLI start-up. Variables that
are already set in the shell take precedence.

| Variable | Effect |
| --- | --- |
| `IPS_R_ENV` | Deployment context label |
| `IPS_R_LOG_LEVEL`, `IPS_R_LOG_FORMAT` | Default logging (`logging.*`) |
| `IPS_R_SEED` | Default `seed.value` |
| `IPS_R_ROOT` | Project root (else auto-detected) |
| `IPS_R_CONFIG_DIR` | Config directory (else `<root>/configs`) |
| `IPS_R_DATA_DIR`, `IPS_R_CACHE_DIR`, `IPS_R_MODELS_DIR`, `IPS_R_OUTPUT_DIR` | Redirect working directories |

Precedence, from highest to lowest: CLI override → shell env → `.env` → YAML default.
