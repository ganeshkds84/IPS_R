"""End-to-end tests of the ``ips-r`` command-line interface."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ips_r import __version__
from ips_r.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_and_components() -> None:
    assert runner.invoke(app, ["info"]).exit_code == 0
    result = runner.invoke(app, ["components"])
    assert result.exit_code == 0
    assert "retriever" in result.stdout


def test_config_with_overrides() -> None:
    result = runner.invoke(app, ["config", "seed.value=31"])
    assert result.exit_code == 0, result.output
    assert "value: 31" in result.stdout


def test_config_error_exits_nonzero() -> None:
    result = runner.invoke(app, ["config", "bogus=1"])
    assert result.exit_code == 1


def test_run_dry_run(tmp_path: Path) -> None:
    root = tmp_path.as_posix()
    result = runner.invoke(app, ["run", "--dry-run", f"paths.root='{root}'", "experiment=example"])
    assert result.exit_code == 0, result.output
    assert "[dry_run]" in result.stdout
    assert any((tmp_path / "outputs" / "example").iterdir())


def test_missing_env_file_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--env-file", str(tmp_path / "nope.env"), "version"])
    assert result.exit_code == 1
