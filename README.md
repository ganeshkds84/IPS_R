# IPS-R

Research codebase for the IPS-R M.Tech thesis project.

**Status: Sprint 1 (Infrastructure).** This repository provides reproducible
experiment infrastructure and the abstract interfaces for every research
component. It contains **no research algorithms** yet; they arrive in later
sprints as implementations of the interfaces below.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
# Linux / macOS / Git Bash
./scripts/bootstrap.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

Or manually:

```bash
uv python install 3.12
uv sync                       # creates .venv with runtime + dev dependencies
cp .env.example .env          # optional
uv run ips-r info
uv run ips-r run --dry-run experiment=example
```

## Command-line interface

| Command | Purpose |
| --- | --- |
| `ips-r info` | Python, platform, project root and config directory |
| `ips-r config [overrides…]` | Print the fully composed, resolved config |
| `ips-r components` | List registered implementations per component kind |
| `ips-r run [overrides…] [--dry-run]` | Create a tracked run and execute the configured pipeline |
| `ips-r version` | Package version |

Global options: `--env-file PATH`, `--verbose/-v`. `python -m ips_r` is equivalent to `ips-r`.

Overrides use [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic/):

```bash
uv run ips-r run experiment=example seed.value=7 logging.level=DEBUG
uv run ips-r run retriever=my_retriever +retriever.params.top_k=50
```

## Repository layout

```
├── src/ips_r/             Python package (src layout)
│   ├── core/              exceptions, logging, env, paths, seeds, registry, data types
│   ├── config/            structured config schema + Hydra compose loader
│   ├── components/        AbstractComponent (lifecycle, guarded execution)
│   ├── data/              AbstractDataset
│   ├── retrieval/         AbstractRetriever
│   ├── candidates/        AbstractCandidateGenerator
│   ├── policy/            AbstractDecisionPolicy
│   ├── reranking/         AbstractReranker
│   ├── generation/        AbstractGenerator
│   ├── evaluation/        AbstractEvaluator
│   ├── pipeline/          AbstractPipeline + config-driven builder
│   ├── experiment/        run directories, metadata, execution
│   └── cli.py             `ips-r` entrypoint (Typer)
├── configs/               Hydra config groups (one folder per component kind)
├── experiments/           experiment plans and notes (presets live in configs/experiment/)
├── scripts/               bootstrap and helper scripts
├── tests/                 unit + integration tests (pytest)
├── docs/                  project documentation
├── notebooks/             exploratory notebooks
├── data/                  datasets (raw / interim / processed / external) — git-ignored
├── models/                model checkpoints — git-ignored
├── cache/                 caches (HF, embeddings, …) — git-ignored
└── outputs/               run directories — git-ignored
```

## Every run is reproducible

`ips-r run` creates `outputs/<experiment>/<run_id>/` containing:

- `config.yaml` — the fully resolved configuration
- `metadata.json` — git commit (with `-dirty` flag), Python, platform, seed, status, timing
- `run.log` — JSON-lines structured log
- `results.json` — pipeline summary (when a pipeline runs)

Seeds are applied to `random`, NumPy and (if installed) PyTorch. See
[docs/reproducibility.md](docs/reproducibility.md).

## Development

```bash
make format      # ruff --fix + black
make lint        # ruff + black --check
make typecheck   # mypy --strict
make test        # pytest
make check       # everything CI runs
```

Without `make`, run the underlying commands directly, e.g. `uv run pytest`.
Pre-commit hooks are installed by `make install` / the bootstrap scripts (requires git).
See [docs/development.md](docs/development.md).

## Docker

```bash
docker build -t ips-r .
docker run --rm ips-r run --dry-run experiment=example
docker compose run --rm ips-r info      # mounts data/, models/, cache/, outputs/
```

## Documentation

- [Architecture & interfaces](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Reproducibility](docs/reproducibility.md)
- [Development workflow](docs/development.md)

## License

[MIT](LICENSE)
