from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cartsy_dedupe.cli import resolve_run_output_dir


def test_run_output_dir_is_scoped_to_timestamped_run_id() -> None:
    now = datetime(2026, 4, 30, 15, 4, 5)
    assert resolve_run_output_dir(Path("outputs"), now=now) == Path("outputs/run_20260430_150405")


def test_run_output_dir_is_not_double_nested_for_existing_run_id() -> None:
    assert resolve_run_output_dir(Path("outputs/run_20260430_150405")) == Path("outputs/run_20260430_150405")
