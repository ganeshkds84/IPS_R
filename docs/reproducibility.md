# Reproducibility

## Seeds

`ips_r.core.seed.seed_everything(seed, deterministic=False)` seeds:

- Python `random`
- NumPy's legacy global RNG
- PyTorch CPU and CUDA, if torch is installed; with `deterministic=True` it also
  enables deterministic algorithms and cuDNN determinism
- `PYTHONHASHSEED`, which affects **child** processes only

Prefer explicit randomness over global state:

```python
from ips_r.core.seed import derive_seed, make_rng

rng = make_rng(derive_seed(cfg.seed.value, "retriever", fold))
```

`derive_seed` is a SHA-256 based derivation. It does not depend on Python's hash
randomisation, so child seeds are stable across processes and machines.

## Run directories

Each `ips-r run` writes `outputs/<experiment>/<YYYYmmdd-HHMMSS>-<hex>/`:

| File | Contents |
| --- | --- |
| `config.yaml` | Fully resolved config; re-run with `ips-r run` plus the same overrides |
| `metadata.json` | `status` (`running` / `completed` / `failed`), timings, git commit (`-dirty` when there are uncommitted changes), Python and platform, argv, seed state, resolved paths, error message on failure |
| `run.log` | JSON-lines log of the run |
| `results.json` | Pipeline summary, or preparation summary for `ips-r prepare` |
| `dataset/` | Prepared dataset (`ips-r prepare` only); see below |
| `knowledge/` | Knowledge artifacts (`ips-r build-knowledge` only); see below |

## Prepared datasets

Dataset preparation is deterministic for a given config and set of input files:

- adapters derive ids from source data, such as the SQuAD title plus paragraph index,
  the HotpotQA Wikipedia title, the NQ `example_id`, the MS MARCO `pid` or the PDF
  file stem;
- PDF collections are processed in sorted relative-path order;
- chunk ids are `<document_id>::<index>`;
- JSON outputs are written with sorted keys.

`dataset/metadata.json` records a `sha256:` **content hash** over every accepted
document and question, in stream order. If two runs report the same hash, they
processed identical data. The metadata also records the chunker and its parameters.

## Knowledge artifacts

`ips-r build-knowledge` is deterministic for a given config and set of input files.
Rebuilding produces byte-identical data files, manifest and checksums:

- object ids derive from the document id and the reading-order position
  (`<document_id>#s0003`); chunk ids hash the parent unit, the unit span and the text
  (`<document_id>::<16 hex>`);
- documents are processed in dataset stream order, and objects in reading order;
- JSON records use sorted keys; Arrow and Parquet schemas are fixed per table, and
  nested values are stored as canonical JSON strings;
- `manifest.json` and `checksums.sha256` contain no timestamps. Wall-clock timings
  live only in `reports/`, which the manifest and checksums do not cover.

Byte-identical Arrow and Parquet files also need the same pyarrow version, which the
manifest records (`pyarrow_version`, next to `package_version`).

### Hashes and invalidation

| Hash | Covers |
| --- | --- |
| `KnowledgeDocument.source_hash` | Every content field of the source document except `source` (its file path) |
| Processing fingerprints | The normalisation, structure, chunking (including the `chunker=` strategy and its params) and enrichment settings, plus the schema version |
| `KnowledgeDocument.knowledge_hash` | Source hash + fingerprints + every derived object |
| `KnowledgeChunk.content_hash` / `lineage_hash` | Chunk text / text + source hash + fingerprints |
| `artifact_hash` (manifest) | The ordered per-document hashes + fingerprints + schema version |

Editing one document changes that document's hash and the artifact hash. The
other documents keep their hashes, and
`ips_r.knowledge.diff_document_hashes(old, new)` lists exactly what to rebuild.
Changing any processing setting changes every hash. Moving the data directory
changes none. An `IndexArtifact` records the artifact hash it was built from, so
`is_stale()` detects an index built from older knowledge.

### Verification

```bash
uv run ips-r verify-artifacts outputs/<experiment>/<run_id>/knowledge
cd outputs/<experiment>/<run_id>/knowledge && sha256sum -c checksums.sha256
```

`verify-artifacts` checks the schema version and re-hashes every file listed in the
manifest and in `checksums.sha256`. It exits non-zero and lists each problem
(missing file, checksum mismatch).

## Dependencies

`uv.lock` pins every transitive dependency. CI and Docker install with
`uv sync --frozen`, so they fail if the lock file is out of date.
