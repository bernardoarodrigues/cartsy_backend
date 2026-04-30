from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

try:
    import duckdb
except ImportError:  # pragma: no cover - fallback for minimal environments.
    duckdb = None


def iter_csv_rows(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            if limit is not None and idx > limit:
                break
            yield {key: value or "" for key, value in row.items()}


def load_rows(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    if duckdb is None:
        return list(iter_csv_rows(path, limit=limit))

    query = "SELECT * FROM read_csv_auto(?, header = true, all_varchar = true)"
    params: list[object] = [str(path)]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with duckdb.connect(database=":memory:") as conn:
        rows = conn.execute(query, params).fetchall()
        columns = [column[0] for column in conn.description]
    return [
        {column: "" if value is None else str(value) for column, value in zip(columns, row, strict=True)}
        for row in rows
    ]
