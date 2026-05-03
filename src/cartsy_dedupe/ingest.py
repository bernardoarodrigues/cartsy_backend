from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

try:
    import polars as pl
except ImportError:  # pragma: no cover - fallback for minimal environments.
    pl = None


def iter_csv_rows(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, str]]:
    """Yield CSV rows as string dictionaries with empty-string null handling."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            if limit is not None and idx > limit:
                break
            yield {key: value or "" for key, value in row.items()}


def load_rows(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load CSV rows with Polars when available and csv as a fallback."""
    if pl is None:
        return list(iter_csv_rows(path, limit=limit))

    frame = pl.read_csv(
        str(path),
        infer_schema_length=0,
        null_values=[],
        truncate_ragged_lines=True,
    )
    if limit is not None:
        frame = frame.head(limit)
    columns = frame.columns
    rows = frame.iter_rows()
    return [
        {column: "" if value is None else str(value) for column, value in zip(columns, row, strict=True)}
        for row in rows
    ]
