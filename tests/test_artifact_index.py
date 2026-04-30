from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl

from cartsy_dedupe.artifact_index import (
    artifact_search_sql,
    build_artifact_documents,
    embedding_text_for_artifact,
)


def make_artifact_run_dir(tmp_path: Path) -> Path:
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


def test_build_artifact_documents_indexes_groups_offers_and_pairs(tmp_path: Path) -> None:
    run_dir = make_artifact_run_dir(tmp_path)

    docs = build_artifact_documents(run_dir)
    by_id = {doc.artifact_id: doc for doc in docs}

    assert "group:prod_abc" in by_id
    assert "offer:1" in by_id
    assert "pair:1:2" in by_id
    assert by_id["group:prod_abc"].metadata["graph"]["offers"] == ["offer:1", "offer:2"]
    assert by_id["offer:1"].metadata["graph"]["group"] == "group:prod_abc"
    assert by_id["pair:1:2"].metadata["graph"]["offers"] == ["offer:1", "offer:2"]


def test_artifact_documents_include_semantic_evidence_text(tmp_path: Path) -> None:
    run_dir = make_artifact_run_dir(tmp_path)
    docs = build_artifact_documents(run_dir)
    group = next(doc for doc in docs if doc.artifact_id == "group:prod_abc")
    pair = next(doc for doc in docs if doc.artifact_id == "pair:1:2")

    assert "cetaphil locao hidratante" in group.search_text
    assert "brand match" in pair.search_text
    assert "type: group" in embedding_text_for_artifact(group)


def test_artifact_search_sql_toggles_vector_evidence() -> None:
    lexical_sql = artifact_search_sql(include_vector=False)
    vector_sql = artifact_search_sql(include_vector=True)

    assert "embedding <=>" not in lexical_sql
    assert "embedding <=>" in vector_sql
    assert "backend:artifact_index" in vector_sql
