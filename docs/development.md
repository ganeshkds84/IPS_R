# Development workflow

## Tooling

| Tool | Role | Config |
| --- | --- | --- |
| uv | Python 3.12, virtualenv, dependency locking | `pyproject.toml`, `uv.lock`, `.python-version` |
| Ruff | Linting (pycodestyle, pyflakes, isort, bugbear, pydocstyle/Google, pylint, …) | `[tool.ruff]` |
| Black | Formatting (line length 100) | `[tool.black]` |
| Mypy | `--strict` type checking of `src/`, `tests/`, `scripts/` | `[tool.mypy]` |
| pytest | Tests; warnings are errors; coverage gate ≥ 85 % | `[tool.pytest.ini_options]`, `[tool.coverage.*]` |
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

Every module must have:

- a module docstring, plus Google-style docstrings on public classes and functions (enforced by Ruff `D`)
- full type hints (enforced by mypy `--strict`)
- a module-level `logger = get_logger(__name__)` wherever the module does work
- errors raised as subclasses of `IPSRError`, never bare `Exception`

## Tests

- `tests/unit/`: isolated tests of each module
- `tests/integration/`: config → components → run directory, and the CLI
- `tests/fakes.py`: minimal in-memory implementations of every interface
- The autouse fixture in `tests/conftest.py` strips `IPS_R_*` variables and resets logging between tests.

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
