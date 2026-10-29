# AGENTS.md: engineering rules for IPS-R

These rules apply to every contributor, human or AI agent. They are binding. A change
that breaks a rule is not finished until the rule is satisfied, or until this file is
changed in the same pull request with a stated reason.

The research design (methodology, architecture, component boundaries) is **frozen**.
Engineering work implements it and does not redesign it. If an implementation reveals a
critical design problem, stop and raise it. Do not work around it silently.

---

## 1. Architecture principles

- **Layering.** Dependencies point downward only:

  ```
  cli ─► experiment ─► pipeline ─► component packages ─► components ─► core
                                   (data, retrieval, …)      config.schema ─► core
  ```

  `core` imports nothing from the rest of the package. A component package never imports
  `cli`, `experiment` or `pipeline`. Component packages do not import each other. The one
  exception is the data contracts in `ips_r.data.models` and `ips_r.core.types`.
- **Interfaces first.** Every pluggable capability is an `AbstractComponent` subclass
  with a public template method (validation, logging, contracts) and a protected
  `_hook` that implementations override. Implementations contain domain logic only.
- **Registries, not imports.** Implementations register themselves under a short name
  (`@DATASETS.register("squad")`) and are chosen through Hydra config. The package
  `__init__` must import implementation modules so that registration runs.
- **Immutable data contracts.** Values passed between components are frozen dataclasses
  that validate in `__post_init__`. Free-form extras go into a read-only
  `metadata`/`extra` mapping, never into new positional fields.
- **Streaming by default.** Code that can see a full corpus (datasets, validation,
  statistics, chunking) must work on iterators. Materialising a corpus into a list is
  opt-in.
- **Clean architecture.** Pure logic (layout analysis, chunk splitting, validation rules,
  statistics) is kept apart from I/O (file reading, PDF backends, report writing), so
  the logic can be tested without files.

## 2. SOLID in this codebase

| Principle | How it applies here |
| --- | --- |
| Single responsibility | A module does one job: `io.py` reads, `validation.py` validates, `statistics.py` measures. |
| Open/closed | New datasets, chunkers and PDF backends are added by registering a class, never by editing `if/elif` chains. |
| Liskov | A subclass honours its base contract (e.g. chunk text equals `document.text[start:end]`). Base classes enforce contracts at runtime. |
| Interface segregation | Hooks are small (`_split(text) -> spans`, `_iter_documents()`), and optional hooks have no-op defaults. |
| Dependency inversion | High-level services (`prepare_dataset`) depend on abstractions (`AbstractDataset`, `AbstractChunker`, `AbstractPDFBackend`), which are injected. |

## 3. Coding standards

- Python 3.12 only. Use modern syntax: PEP 695 generics, `X | None`, `match` where it reads better.
- `from __future__ import annotations` in every module **except** `config/schema.py`,
  where OmegaConf inspects real annotations.
- Ruff (rule set in `pyproject.toml`) and Black (100 columns) must pass with no warnings.
  Never add a blanket `# noqa`. A targeted `# noqa: CODE - reason` is allowed when it is justified.
- Prefer `pathlib.Path` and keyword-only arguments for options.
  Mutable default arguments are banned.
- Keep functions short. Extract a helper once a function needs a comment to explain a block.

## 4. Naming conventions

| Element | Convention | Example |
| --- | --- | --- |
| Modules, packages | `snake_case` | `sliding_window.py` |
| Classes | `PascalCase`; abstract bases start with `Abstract` | `AbstractChunker` |
| Registries | `UPPER_SNAKE` plural | `CHUNKERS`, `DATASETS` |
| Registry names / config files | `snake_case`, identical to each other | `recursive_character` ↔ `configs/chunker/recursive_character.yaml` |
| Implementation hooks | leading underscore | `_split`, `_iter_documents` |
| Log events | dot-namespaced, lower case, `noun.verb` | `dataset.documents.iterated` |
| Constants | `UPPER_SNAKE` with `Final` | `DEFAULT_SEPARATORS: Final` |
| Tests | `test_<behaviour>` | `test_overlap_is_respected` |

## 5. Type hints

- `mypy --strict` must pass over `src/`, `tests/` and `scripts/`.
- Every function, including private helpers and test functions, is fully annotated.
- `Any` is allowed only at true boundaries: raw JSON records and Hydra `params`.
  Narrow it to a concrete type straight away.
- Do not add `# type: ignore` without an error code and a reason.

## 6. Logging requirements

- Every module that does work defines `logger = get_logger(__name__)`. Components use
  `self._logger`, which is bound to the component name and type.
- Log **structured events** with key/value fields (`logger.info("chunker.done", chunks=12)`),
  never f-string sentences.
- Levels: `debug` for per-record detail, `info` for one line per stage, `warning` for
  recoverable data problems, `error` for failures. Never log inside tight per-token loops.
- Never use `print` in library code. Only the CLI writes to stdout, through Typer/Rich.

## 7. Hydra configuration rules

- Every pluggable component has one config group, `configs/<kind>/`, and one file per
  implementation. The file shape is always `name: <registry name>` + `params: {...}`.
- `none.yaml` disables a component (`name: null`) and is the default.
- Non-component settings get a structured dataclass in `ips_r.config.schema`, so unknown
  keys and type errors fail when the config is composed.
- **No hardcoded values.** Thresholds, sizes, paths, patterns and limits belong in YAML.
  A code default must equal its YAML default. It exists only so the class is usable in
  tests and notebooks.
- Paths in configs are built from `${paths.*}` interpolations, never absolute literals.
- Validate parameters in the component constructor (`int_param`, `float_param`, …) so a
  misconfiguration fails at build time (`ips-r run --dry-run`), not mid-run.

## 8. Reproducibility requirements

- Every executable entry point runs inside `run_context`, which snapshots the config,
  records metadata (git commit, versions, seed) and writes logs.
- Randomness only comes from `make_rng(derive_seed(cfg.seed.value, ...))`.
  Global RNG state is banned in library code.
- Outputs must be deterministic for a given config and input: sort where iteration order
  is not guaranteed, and use stable IDs derived from source data, never counters shared
  across runs or hashes salted by `PYTHONHASHSEED`.
- Prepared data records a content hash of what was processed.
- Dependencies are pinned in `uv.lock`. Add them with `uv add`; never edit the lock by hand.

## 9. Error handling philosophy

- Errors raised on purpose derive from `IPSRError`. Pick the narrowest existing subclass
  (`ConfigurationError`, `ComponentError`, `DataError`, `PDFParseError`, …). Add a new
  subclass only when callers need to tell it apart.
- **Fail fast on programmer and configuration errors. Degrade gracefully on data errors.**
  A malformed record is reported as a `ValidationIssue` and skipped. It aborts processing
  only when the user opts in (`strict: true`, `fail_on_error: true`).
- Never swallow an exception silently. When catching broadly at an I/O boundary, convert
  the exception, chain it (`raise ... from exc`) and log it.
- Public methods document the exceptions they raise in a Google-style `Raises:` section.

## 10. Modular design requirements

- One concept per module. Keep modules under ~400 lines and split them before they grow
  past that.
- Third-party libraries that are heavy or optional (PDF engines, ML frameworks) are imported
  only inside the single adapter module that wraps them, and lazily where possible.
- Public API is re-exported from the package `__init__` with an explicit `__all__`.

## 11. Testing requirements

- Total coverage must stay **≥ 95 %**, enforced by `[tool.coverage.report] fail_under`.
- Pure logic gets unit tests (`tests/unit/`). Config → components → artifacts flows and
  the CLI get integration tests (`tests/integration/`).
- Tests are hermetic. They need no network, write only to `tmp_path`, and build fixtures
  in code (`tests/data_factories.py`). Binary fixtures must not be committed.
- Every bug fix comes with a test that fails without the fix.
- Warnings are errors (`filterwarnings = ["error"]`). Fix the cause; do not filter it.

## 12. Documentation standards

- Module docstring plus Google-style docstrings on every public class and function
  (enforced by Ruff `D`).
- When behaviour changes, update `README.md` and `docs/` in the same change.
  **Only document what is actually implemented.**
- Each config file starts with a comment saying what it selects and how to override it.

## 13. Dependency management

- Runtime dependencies are added with `uv add`, dev-only ones with `uv add --group dev`.
  Commit both `pyproject.toml` and `uv.lock`.
- Prefer permissive licences (MIT/BSD/Apache). Copyleft dependencies (GPL/AGPL) are not allowed.
- Every new dependency needs a one-line reason in the pull request description.
- CI and Docker install with `uv sync --frozen`.

## 14. Definition of done (every change)

```bash
uv run ruff check . && uv run black --check . && uv run mypy && uv run pytest --cov
```

All four pass. Docs are updated. No new `TODO` is left without an owner and a milestone.
