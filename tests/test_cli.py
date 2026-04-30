from __future__ import annotations

from pathlib import Path

from cartsy_dedupe.cli import resolve_run_output_dir


def test_run_output_dir_is_scoped_to_default_run_name() -> None:
    assert resolve_run_output_dir(Path("outputs")) == Path("outputs/run_postgres_openai")


def test_run_output_dir_is_not_double_nested() -> None:
    assert resolve_run_output_dir(Path("outputs/run_postgres_openai")) == Path("outputs/run_postgres_openai")
