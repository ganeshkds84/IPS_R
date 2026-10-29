"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ips_r.core.exceptions import ConfigurationError
from ips_r.core.log import (
    bind_context,
    configure_logging,
    get_logger,
    remove_file_handlers,
)


def test_json_file_receives_structlog_and_stdlib_records(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    configure_logging("INFO", "json", log_file)
    bind_context(run_id="r1")
    get_logger("test").info("hello", answer=42)
    logging.getLogger("thirdparty").warning("plain %s", "stdlib")
    remove_file_handlers()

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    hello = next(line for line in lines if line["event"] == "hello")
    assert hello["answer"] == 42
    assert hello["run_id"] == "r1"
    assert hello["level"] == "info"
    assert any(line["event"] == "plain stdlib" for line in lines)


def test_level_filters_records(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    configure_logging("warning", "console", log_file)
    get_logger("test").info("hidden")
    get_logger("test").error("shown")
    remove_file_handlers()
    content = log_file.read_text(encoding="utf-8")
    assert "shown" in content
    assert "hidden" not in content


@pytest.mark.parametrize(("level", "fmt"), [("LOUD", "console"), ("INFO", "xml")])
def test_invalid_settings_raise(level: str, fmt: str) -> None:
    with pytest.raises(ConfigurationError):
        configure_logging(level, fmt)


def test_reconfiguring_replaces_handlers() -> None:
    configure_logging("INFO")
    configure_logging("DEBUG")
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_ips_r_handler", False)]
    assert len(ours) == 1
