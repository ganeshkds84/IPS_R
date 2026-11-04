"""Command-line interface (``ips-r``).

Examples::

    ips-r info
    ips-r config seed.value=7 experiment=example
    ips-r run --dry-run
    ips-r prepare dataset=squad chunker=recursive_character
    ips-r build-knowledge dataset=academic_pdf chunker=sentence
    ips-r verify-artifacts outputs/default/<run_id>/knowledge
    ips-r --env-file .env.local run logging.level=DEBUG
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from ips_r import __version__
from ips_r.config.loader import load_config, to_yaml
from ips_r.core.env import load_env_file, load_settings
from ips_r.core.exceptions import IPSRError
from ips_r.core.log import configure_logging, get_logger
from ips_r.core.paths import find_config_dir, find_project_root
from ips_r.experiment.runner import (
    execute_knowledge,
    execute_preparation,
    execute_run,
    run_context,
)
from ips_r.knowledge.serialization import verify_artifacts
from ips_r.pipeline.builder import COMPONENT_REGISTRIES

logger = get_logger(__name__)
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="ips-r",
    help="IPS-R research toolkit.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

OverridesArg = Annotated[
    list[str] | None,
    typer.Argument(
        help="Hydra overrides, e.g. seed.value=7 experiment=example.", show_default=False
    ),
]


def _fail(exc: BaseException) -> NoReturn:
    """Report an expected error and exit with status 1."""
    logger.error("command.failed", error=str(exc), error_type=type(exc).__name__)
    err_console.print(f"[bold red]error:[/] {exc}", highlight=False)
    raise typer.Exit(code=1)


@app.callback()
def bootstrap(
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", help="Load environment variables from this file."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Load the environment and configure baseline logging before any command."""
    try:
        configure_logging(level="DEBUG" if verbose else "INFO")
        load_env_file(env_file, required=env_file is not None)
        settings = load_settings()
        configure_logging(level="DEBUG" if verbose else settings.log_level, fmt=settings.log_format)
    except IPSRError as exc:
        _fail(exc)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def info() -> None:
    """Show environment and project information."""
    table = Table(title="IPS-R environment", show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("version", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    try:
        root = find_project_root()
        table.add_row("project root", str(root))
        table.add_row("config dir", str(find_config_dir(root)))
    except IPSRError as exc:
        table.add_row("config dir", f"[red]{exc}[/]")
    console.print(table)


@app.command("config")
def show_config(overrides: OverridesArg = None) -> None:
    """Print the fully composed and resolved configuration."""
    try:
        cfg = load_config(overrides or [])
        typer.echo(to_yaml(cfg))
    except IPSRError as exc:
        _fail(exc)


@app.command()
def components() -> None:
    """List registered component implementations."""
    table = Table(title="Registered components")
    table.add_column("kind", style="bold")
    table.add_column("implementations")
    for kind, registry in COMPONENT_REGISTRIES.items():
        table.add_row(kind, ", ".join(registry.names()) or "[dim]none[/]")
    console.print(table)


@app.command()
def run(
    overrides: OverridesArg = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate config and components without executing.")
    ] = False,
) -> None:
    """Create a tracked run: compose config, seed, snapshot, build components and execute."""
    try:
        cfg = load_config(overrides or [])
        with run_context(cfg) as ctx:
            summary = execute_run(ctx, dry_run=dry_run)
        typer.echo(f"run {ctx.run_id} [{summary['status']}] -> {ctx.output_dir}")
    except IPSRError as exc:
        _fail(exc)


@app.command()
def prepare(overrides: OverridesArg = None) -> None:
    """Ingest, validate, chunk and profile the configured dataset in a tracked run."""
    try:
        cfg = load_config(overrides or [])
        with run_context(cfg) as ctx:
            summary = execute_preparation(ctx)
    except IPSRError as exc:
        _fail(exc)
    table = Table(title=f"Prepared dataset: {summary['dataset']}", show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    validation = summary["validation"]
    for key, value in (
        ("documents", summary["num_documents"]),
        ("questions", summary["num_questions"]),
        ("chunks", summary["num_chunks"]),
        ("pages", summary["num_pages"]),
        (
            "dropped",
            f"{summary['dropped_documents']} documents, {summary['dropped_questions']} questions",
        ),
        ("validation", f"{validation['errors']} errors, {validation['warnings']} warnings"),
        ("output", summary["output_dir"]),
    ):
        table.add_row(key, str(value))
    console.print(table)
    typer.echo(f"run {ctx.run_id} [{summary['status']}] -> {ctx.output_dir}")


@app.command("build-knowledge")
def build_knowledge(overrides: OverridesArg = None) -> None:
    """Build index-ready knowledge artifacts for the configured dataset in a tracked run."""
    try:
        cfg = load_config(overrides or [])
        with run_context(cfg) as ctx:
            summary = execute_knowledge(ctx)
    except IPSRError as exc:
        _fail(exc)
    table = Table(title="Knowledge artifacts", show_header=False)
    table.add_column("key", style="bold")
    table.add_column("value")
    for key in ("artifact_id", "documents", "sections", "chunks", "citations", "references"):
        table.add_row(key, str(summary[key]))
    table.add_row(
        "dropped / failed", f"{summary['documents_dropped']} / {summary['documents_failed']}"
    )
    table.add_row("output", summary["output_dir"])
    console.print(table)
    typer.echo(f"run {ctx.run_id} [{summary['status']}] -> {ctx.output_dir}")


@app.command("verify-artifacts")
def verify(
    directory: Annotated[Path, typer.Argument(help="Knowledge artifact directory.")],
) -> None:
    """Verify an artifact directory against its manifest and checksums."""
    try:
        problems = verify_artifacts(directory)
    except IPSRError as exc:
        _fail(exc)
    if problems:
        for problem in problems:
            err_console.print(f"[red]{problem}[/]", highlight=False)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {directory}")


def main() -> None:
    """Console-script entrypoint."""
    app()
