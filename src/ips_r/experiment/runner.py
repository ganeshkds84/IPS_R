"""Run lifecycle: every invocation gets an isolated, fully-described output directory.

Layout of a run directory::

    outputs/<experiment>/<run_id>/
        config.yaml     # fully resolved configuration
        metadata.json   # environment, git commit, seed, status, timing
        run.log         # JSON-lines log (if logging.to_file)
        results.json    # pipeline / preparation summary
        dataset/        # prepared dataset artefacts (``ips-r prepare``)
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ips_r import __version__
from ips_r.config.loader import as_app_config, save_config
from ips_r.core.exceptions import ConfigurationError, IPSRError
from ips_r.core.log import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    remove_file_handlers,
)
from ips_r.core.paths import ProjectPaths
from ips_r.core.seed import SeedState, seed_everything
from ips_r.data import CHUNKERS, DATASETS, prepare_dataset
from ips_r.pipeline.builder import build_component, build_pipeline, build_pipeline_components

logger = get_logger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything that identifies and describes a single run."""

    run_id: str
    experiment: str
    output_dir: Path
    config: DictConfig
    paths: ProjectPaths
    seed: SeedState
    started_at: datetime

    @property
    def metadata_path(self) -> Path:
        """Location of ``metadata.json``."""
        return self.output_dir / "metadata.json"


def make_run_id(now: datetime | None = None) -> str:
    """Return a sortable, collision-resistant run id such as ``20260101-120000-a1b2c3``."""
    moment = now or datetime.now(UTC)
    return f"{moment:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def safe_name(name: str) -> str:
    """Make ``name`` safe for use as a directory name."""
    return _SAFE_NAME.sub("-", name).strip("-.") or "default"


def git_commit(cwd: Path) -> str | None:
    """Return the current git commit hash (with ``-dirty`` suffix), or ``None`` if unavailable."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git.unavailable", error=repr(exc))
        return None
    commit = head.stdout.strip()
    return f"{commit}-dirty" if status.stdout.strip() else commit


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as pretty-printed JSON.

    Raises:
        ConfigurationError: If the file cannot be written.
    """
    try:
        path.write_text(
            json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        raise ConfigurationError(f"Cannot write {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``, returning ``{}`` if missing or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("run.metadata.unreadable", path=str(path), error=repr(exc))
        return {}
    return data if isinstance(data, dict) else {}


def start_run(cfg: DictConfig) -> RunContext:
    """Prepare a run: directories, logging, seeding, config snapshot and metadata.

    Raises:
        ConfigurationError: If directories or files cannot be created or config is invalid.
        SeedError: If the configured seed is invalid.
    """
    started_at = datetime.now(UTC)
    run_id = make_run_id(started_at)
    experiment = safe_name(str(cfg.experiment.name))
    paths = ProjectPaths.from_mapping(cfg.paths)
    paths.ensure()

    output_dir = (
        Path(str(cfg.run.output_dir)).expanduser().resolve()
        if cfg.run.output_dir
        else paths.outputs / experiment / run_id
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"Cannot create run directory {output_dir}: {exc}") from exc

    configure_logging(
        level=str(cfg.logging.level),
        fmt=str(cfg.logging.format),
        log_file=output_dir / "run.log" if cfg.logging.to_file else None,
    )
    bind_context(run_id=run_id, experiment=experiment)

    seed_state = seed_everything(int(cfg.seed.value), deterministic=bool(cfg.seed.deterministic))
    save_config(cfg, output_dir / "config.yaml")

    ctx = RunContext(
        run_id=run_id,
        experiment=experiment,
        output_dir=output_dir,
        config=cfg,
        paths=paths,
        seed=seed_state,
        started_at=started_at,
    )
    _write_json(
        ctx.metadata_path,
        {
            "run_id": run_id,
            "experiment": experiment,
            "status": "running",
            "started_at": started_at.isoformat(),
            "package_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "argv": sys.argv,
            "git_commit": git_commit(paths.root),
            "seed": asdict(seed_state),
            "paths": paths.as_dict(),
        },
    )
    logger.info("run.started", output_dir=str(output_dir))
    return ctx


def finalize_run(
    ctx: RunContext, status: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Record the final status and timing in ``metadata.json``.

    Returns:
        The updated metadata.
    """
    finished_at = datetime.now(UTC)
    metadata = _read_json(ctx.metadata_path)
    metadata.update(
        {
            "status": status,
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - ctx.started_at).total_seconds(), 3),
        }
    )
    if extra:
        metadata.update(extra)
    try:
        _write_json(ctx.metadata_path, metadata)
    except IPSRError as exc:
        logger.error("run.metadata.write_failed", error=str(exc))
    logger.info("run.finished", status=status, duration_seconds=metadata["duration_seconds"])
    return metadata


@contextmanager
def run_context(cfg: DictConfig) -> Iterator[RunContext]:
    """Context manager wrapping :func:`start_run` / :func:`finalize_run`.

    The run is marked ``failed`` (with the error message) if the body raises,
    and the log-file handler is always closed on exit.
    """
    ctx = start_run(cfg)
    try:
        yield ctx
    except BaseException as exc:
        finalize_run(ctx, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    else:
        finalize_run(ctx, "completed")
    finally:
        clear_context()
        remove_file_handlers()


def execute_run(ctx: RunContext, *, dry_run: bool = False) -> dict[str, Any]:
    """Build the configured components and, if a pipeline is configured, run it.

    Args:
        ctx: The active run.
        dry_run: Only validate that components can be constructed.

    Returns:
        A summary dict; also written to ``results.json`` when a pipeline ran.

    Raises:
        ConfigurationError: If a component selection is invalid.
        PipelineError: If construction fails.
        ComponentError: If the pipeline fails.
    """
    components = build_pipeline_components(ctx.config)
    configured = components.configured()
    logger.info("run.components", components=configured)

    if dry_run or ctx.config.run.dry_run:
        logger.info("run.dry_run", components=configured)
        return {"status": "dry_run", "components": configured}

    pipeline = build_pipeline(ctx.config, components)
    if pipeline is None:
        logger.warning(
            "run.no_pipeline",
            hint="No pipeline configured; register one and select it with pipeline=<name>.",
        )
        return {"status": "no_pipeline", "components": configured}

    with pipeline:
        results = pipeline.run()
    _write_json(ctx.output_dir / "results.json", results)
    return {"status": "completed", "components": configured, "results": results}


def execute_preparation(ctx: RunContext) -> dict[str, Any]:
    """Prepare the configured dataset (and chunks) into ``<run_dir>/<preparation.output_subdir>``.

    Builds the ``dataset`` and optional ``chunker`` components from the run config and
    runs :func:`ips_r.data.preparation.prepare_dataset`. The summary is also written to
    ``results.json``.

    Returns:
        The preparation summary.

    Raises:
        ConfigurationError: If no dataset is configured or a selection is invalid.
        PipelineError: If a component cannot be constructed.
        DataValidationError: If validation fails and ``preparation.fail_on_error`` is set.
        ComponentError: If the dataset or chunker fails.
    """
    settings = as_app_config(ctx.config).preparation
    dataset = build_component(DATASETS, ctx.config.get("dataset"))
    if dataset is None:
        raise ConfigurationError("Dataset preparation requires a dataset, e.g. dataset=squad")
    chunker = build_component(CHUNKERS, ctx.config.get("chunker"))
    output_dir = ctx.output_dir / safe_name(settings.output_subdir)
    with dataset, chunker if chunker is not None else nullcontext():
        result = prepare_dataset(dataset, output_dir, chunker=chunker, settings=settings)
    summary = {"status": "completed", **result.summary()}
    _write_json(ctx.output_dir / "results.json", summary)
    return summary
