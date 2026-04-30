from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from .normalize import normalize_row
from .schemas import NormalizedProduct


def iter_csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: value or "" for key, value in row.items()}


def load_normalized_products(path: str | Path, limit: int | None = None) -> list[NormalizedProduct]:
    products: list[NormalizedProduct] = []
    for idx, row in enumerate(iter_csv_rows(path), start=1):
        if limit is not None and idx > limit:
            break
        products.append(normalize_row(row))
        if idx % 50_000 == 0:
            print(f"normalized {idx:,} rows")
    return products
