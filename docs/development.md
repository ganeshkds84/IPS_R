# Development workflow

## Tooling

| Tool | Role | Config |
| --- | --- | --- |
| uv | Python 3.12, virtualenv, dependency locking | `pyproject.toml`, `uv.lock`, `.python-version` |
| Ruff | Linting (pycodestyle, pyflakes, isort, bugbear, pydocstyle/Google, pylint, …) | `[tool.ruff]` |
| Black | Formatting (line length 100) | `[tool.black]` |
| Mypy | `--strict` type checking of `src/`, `tests/`, `scripts/` | `[tool.mypy]` |
| pytest | Tests; warnings are errors; coverage gate ≥ 95 % | `[tool.pytest.ini_options]`, `[tool.coverage.*]` |
| pre-commit | Runs hygiene hooks, Ruff, Black and Mypy on commit | `.pre-commit-config.yaml` |

## Daily loop

```bash
uv sync                  # after pulling
make format              # before committing
make check               # lint + typecheck + tests with coverage
```

Adding dependencies:

```bash
uv add <package>             # runtime
uv add --group dev <package> # development only
```

## Code standards

[AGENTS.md](../AGENTS.md) holds the binding engineering rules: architecture,
naming, logging, Hydra, reproducibility, errors, testing and dependencies.
In summary, every module must have:

- a module docstring, plus Google-style docstrings on public classes and functions (enforced by Ruff `D`)
- full type hints (enforced by mypy `--strict`)
- a module-level `logger = get_logger(__name__)` wherever the module does work
- errors raised as subclasses of `IPSRError`, never bare `Exception`

## Tests

- `tests/unit/`: isolated tests of each module
- `tests/integration/`: config → components → run directory, and the CLI
- `tests/fakes.py`: minimal in-memory implementations of every interface
- `tests/knowledge_fakes.py`: in-memory embedding model, cache, vector store, sparse and
  hybrid indexer that exercise the Milestone 2 interfaces (test doubles, not implementations)
- `tests/knowledge_factories.py`: synthetic structured papers (every element type,
  nested and appendix sections, citations, tables) and an in-memory dataset
- `tests/data_factories.py`: builds hermetic dataset fixtures in `tmp_path` for every
  source format: SQuAD, HotpotQA, NQ (gzip and original layout), MS MARCO TSV, and
  PDFs generated with reportlab (paper, two-column, blank, corrupted). Each fixture
  deliberately mixes valid and broken records, so tests exercise issue reporting as
  well as the happy path. Binary fixtures are never committed.

Runtime dependencies added in Milestone 1:
- `pdfplumber` (MIT): PDF text, font and table extraction.

Dev-only dependencies added in Milestone 1:
- `reportlab` (BSD): builds PDF test fixtures.
- `types-reportlab`: type stubs for reportlab.

Runtime dependencies added in Milestone 2:
- `pyarrow` (Apache-2.0): Apache Arrow IPC and Parquet export of knowledge artifacts.
  It is imported lazily and only in `ips_r.knowledge.artifact_schema`. pyarrow ships
  no type stubs, so `[[tool.mypy.overrides]]` sets `ignore_missing_imports` for
  `pyarrow.*`, and the adapter treats the module as `Any` at that boundary.

The autouse fixture in `tests/conftest.py` strips `IPS_R_*` variables and resets
logging between tests.

Knowledge-pipeline tests include regression tests (byte-identical rebuilds,
per-document invalidation, overlap validation of short tail chunks, writer clean-up on
failure) and stress tests (80 synthetic papers, and one document with more than
160 sections).

### Ruff scope

`[tool.ruff] extend-exclude` lists the git-ignored working directories as
root-anchored paths (`./data`, `./outputs`, …). A bare name such as `data` matches
every directory of that name, and it had silently excluded `src/ips_r/data/` from
`ruff check .`. Use the `./` form for any new exclusion.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request:

1. **quality**: Ruff, Black, Mypy and the pre-commit hooks
2. **test**: pytest with coverage plus a CLI smoke test on Ubuntu, Windows and macOS
3. **docker**: image build and a container smoke test

## Docker

The multi-stage `Dockerfile` resolves dependencies with uv from `uv.lock`. The
runtime image runs as a non-root user, defaults to JSON logs and exposes
`data/`, `cache/`, `models/` and `outputs/` as volumes. `docker-compose.yml`
mounts these from the host. It also contains a commented-out GPU reservation block.
