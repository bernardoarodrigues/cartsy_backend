from __future__ import annotations

import csv
import json
from pathlib import Path

from .text import normalize_text

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency is installed in the project venv.
    import difflib

    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FallbackFuzz()


def search_products(run_dir: str | Path, query: str, *, limit: int = 10) -> list[dict[str, object]]:
    query_norm = normalize_text(query)
    rows = read_assignments(run_dir)
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        haystack = " ".join(
            [
                row.get("name_raw", ""),
                row.get("brand_raw", ""),
                row.get("canonical_name", ""),
                row.get("canonical_brand", ""),
                row.get("retailer", ""),
                row.get("sku", ""),
                row.get("dimension", ""),
            ]
        )
        score = float(fuzz.token_set_ratio(query_norm, normalize_text(haystack))) / 100.0
        if query_norm in normalize_text(haystack):
            score = max(score, 0.99)
        scored.append((score, row))

    results = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [
        {
            "score": round(score, 4),
            "source_id": row.get("source_id", ""),
            "dedupe_id": row.get("dedupe_id", ""),
            "retailer": row.get("retailer", ""),
            "name": row.get("name_raw", ""),
            "brand": row.get("brand_raw", ""),
            "price_cents": row.get("price_cents", ""),
            "cluster_confidence": row.get("cluster_confidence", ""),
            "decision": row.get("decision", ""),
        }
        for score, row in results
    ]


def get_group(run_dir: str | Path, dedupe_id: str) -> dict[str, object]:
    group = read_group(run_dir, dedupe_id)
    offers = [row for row in read_assignments(run_dir) if row.get("dedupe_id") == dedupe_id]
    if group is None and not offers:
        raise LookupError(f"No group found for dedupe_id={dedupe_id}")
    if group is None:
        group = {
            "dedupe_id": dedupe_id,
            "source_ids": [offer["source_id"] for offer in offers],
            "canonical_name": offers[0].get("canonical_name", ""),
            "canonical_brand": offers[0].get("canonical_brand", ""),
            "cluster_confidence": offers[0].get("cluster_confidence", ""),
            "num_offers": len(offers),
            "retailers": sorted({offer.get("retailer", "") for offer in offers if offer.get("retailer")}),
        }
    group = dict(group)
    group["offers"] = [
        {
            "source_id": offer.get("source_id", ""),
            "retailer": offer.get("retailer", ""),
            "name": offer.get("name_raw", ""),
            "brand": offer.get("brand_raw", ""),
            "price_cents": offer.get("price_cents", ""),
            "sku": offer.get("sku", ""),
            "dimension": offer.get("dimension", ""),
        }
        for offer in offers
    ]
    return group


def explain_pair(run_dir: str | Path, source_id_a: str, source_id_b: str) -> dict[str, object]:
    left, right = sorted([source_id_a, source_id_b])
    pair = find_candidate_pair(run_dir, left, right)
    assignments = {row["source_id"]: row for row in read_assignments(run_dir)}
    if pair is None:
        return {
            "found": False,
            "product_a": assignments.get(source_id_a, {"source_id": source_id_a}),
            "product_b": assignments.get(source_id_b, {"source_id": source_id_b}),
            "message": "Pair was not generated as a candidate in this run.",
        }
    return {
        "found": True,
        "product_a": assignments.get(source_id_a, {"source_id": source_id_a}),
        "product_b": assignments.get(source_id_b, {"source_id": source_id_b}),
        "pair": pair,
    }


def read_assignments(run_dir: str | Path) -> list[dict[str, str]]:
    path = Path(run_dir) / "product_assignments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_group(run_dir: str | Path, dedupe_id: str) -> dict[str, object] | None:
    path = Path(run_dir) / "dedupe_groups.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            group = json.loads(line)
            if group.get("dedupe_id") == dedupe_id:
                return group
    return None


def find_candidate_pair(run_dir: str | Path, source_id_a: str, source_id_b: str) -> dict[str, object] | None:
    run_path = Path(run_dir)
    parquet_path = run_path / "candidate_pairs.parquet"
    csv_path = run_path / "candidate_pairs.csv"
    if parquet_path.exists():
        try:
            import polars as pl

            df = pl.read_parquet(parquet_path)
            found = df.filter(
                (
                    (pl.col("product_a_id").cast(pl.Utf8) == source_id_a)
                    & (pl.col("product_b_id").cast(pl.Utf8) == source_id_b)
                )
                | (
                    (pl.col("product_a_id").cast(pl.Utf8) == source_id_b)
                    & (pl.col("product_b_id").cast(pl.Utf8) == source_id_a)
                )
            )
            if found.height:
                return found.row(0, named=True)
        except ImportError:
            pass
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pair_ids = {row.get("product_a_id", ""), row.get("product_b_id", "")}
                if pair_ids == {source_id_a, source_id_b}:
                    return row
    return None


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        print("No results.")
        return
    widths = {
        column: min(
            48,
            max(len(column), *(len(_stringify(row.get(column, ""))) for row in rows)),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(
                truncate(_stringify(row.get(column, "")), widths[column]).ljust(widths[column])
                for column in columns
            )
        )


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def _stringify(value: object) -> str:
    return "" if value is None else str(value)
