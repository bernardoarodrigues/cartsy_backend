from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl
import pytest

import cartsy_dedupe.query as query_module
from cartsy_dedupe.query import explain_pair, get_group, search_products


def make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assignments = [
        {
            "source_id": "1",
            "dedupe_id": "prod_abc",
            "retailer": "amazon_br",
            "name_raw": "Cetaphil Loção Hidratante 473ml",
            "brand_raw": "Cetaphil",
            "price_cents": "6790",
            "sku": "A1",
            "dimension": "473ml",
            "canonical_name": "Cetaphil Loção Hidratante 473ml",
            "canonical_brand": "Cetaphil",
            "cluster_confidence": "0.91",
            "decision": "grouped",
            "explanation": "brand_match",
        },
        {
            "source_id": "2",
            "dedupe_id": "prod_abc",
            "retailer": "beleza_na_web",
            "name_raw": "Cetaphil Loção Hidratante 473ml",
            "brand_raw": "Cetaphil",
            "price_cents": "9199",
            "sku": "",
            "dimension": "473ml",
            "canonical_name": "Cetaphil Loção Hidratante 473ml",
            "canonical_brand": "Cetaphil",
            "cluster_confidence": "0.91",
            "decision": "grouped",
            "explanation": "brand_match",
        },
    ]
    with (run_dir / "product_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0].keys()))
        writer.writeheader()
        writer.writerows(assignments)

    group = {
        "dedupe_id": "prod_abc",
        "source_ids": ["1", "2"],
        "canonical_name": "Cetaphil Loção Hidratante 473ml",
        "canonical_brand": "Cetaphil",
        "canonical_category": "Beleza>Pele",
        "cluster_confidence": 0.91,
        "num_offers": 2,
        "retailers": ["amazon_br", "beleza_na_web"],
        "price_min_cents": 6790,
        "price_max_cents": 9199,
        "merge_reasons": ["brand_match; title_high"],
    }
    (run_dir / "dedupe_groups.jsonl").write_text(json.dumps(group) + "\n", encoding="utf-8")
    pl.DataFrame(
        [
            {
                "product_a_id": "1",
                "product_b_id": "2",
                "score": 0.91,
                "decision": "merge",
                "explanation": "brand_match; title_high",
                "blocking_keys": "brand_name",
                "feature_scores": "brand:1.000;title:1.000",
            }
        ]
    ).write_parquet(run_dir / "candidate_pairs.parquet")
    return run_dir


def test_search_products_returns_best_matches(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    results = search_products(run_dir, "cetaphil hidratante", limit=1)
    assert results[0]["dedupe_id"] == "prod_abc"
    assert results[0]["score"] > 0.7


def test_search_products_uses_postgres_backend_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = make_run_dir(tmp_path)

    def fake_postgres_search(run_dir: str | Path, query: str, *, limit: int) -> list[dict[str, object]]:
        return [
            {
                "score": 0.98,
                "source_id": "db-1",
                "dedupe_id": "prod_db",
                "retailer": "postgres",
                "brand": "Cetaphil",
                "price_cents": "1",
                "name": "Cetaphil moisturizing lotion",
            }
        ]

    monkeypatch.setattr(query_module, "search_products_postgres", fake_postgres_search)
    results = search_products(run_dir, "cetaphil moisturizer", limit=1)

    assert results[0]["source_id"] == "db-1"


def test_search_products_auto_falls_back_to_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = make_run_dir(tmp_path)

    def unavailable_postgres_search(run_dir: str | Path, query: str, *, limit: int) -> list[dict[str, object]]:
        raise RuntimeError("postgres unavailable")

    monkeypatch.setattr(query_module, "search_products_postgres", unavailable_postgres_search)
    results = search_products(run_dir, "cetaphil hidratante", limit=1)

    assert results[0]["dedupe_id"] == "prod_abc"


def test_search_products_postgres_backend_surfaces_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = make_run_dir(tmp_path)

    def unavailable_postgres_search(run_dir: str | Path, query: str, *, limit: int) -> list[dict[str, object]]:
        raise RuntimeError("postgres unavailable")

    monkeypatch.setattr(query_module, "search_products_postgres", unavailable_postgres_search)

    with pytest.raises(RuntimeError, match="postgres unavailable"):
        search_products(run_dir, "cetaphil hidratante", limit=1, backend="postgres")


def test_get_group_includes_offers(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    group = get_group(run_dir, "prod_abc")
    assert group["num_offers"] == 2
    assert len(group["offers"]) == 2


def test_explain_pair_reads_candidate_pair(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    explanation = explain_pair(run_dir, "2", "1")
    assert explanation["found"] is True
    assert explanation["pair"]["decision"] == "merge"
