from __future__ import annotations

from cartsy_dedupe.pipeline import (
    RunMetrics,
    embedding_text,
    extracted_attribute_score,
    postgres_retrieval_features,
)


def test_postgres_retrieval_features_parse_evidence_scores() -> None:
    features = postgres_retrieval_features(
        {
            "exact:ean:123",
            "lexical:fts:0.4500",
            "trigram:title:0.8100",
            "vector:cosine:0.9200",
        }
    )

    assert features["exact"] == 1.0
    assert features["lexical"] > 0.60
    assert features["trigram"] == 0.81
    assert features["vector"] == 0.92


def test_embedding_text_uses_available_product_fields() -> None:
    text = embedding_text(brand="Rhode", title="Peptide Lip Tint", color=None, category="Beauty")

    assert "brand: Rhode" in text
    assert "title: Peptide Lip Tint" in text
    assert "color:" not in text


def test_extracted_attributes_detect_same_parent_variant_conflict() -> None:
    score, relation, reasons = extracted_attribute_score(
        {"brand": "Rhode", "product_line": "Peptide Lip Treatment", "variant_name": "Salted Caramel"},
        {"brand": "Rhode", "product_line": "Peptide Lip Treatment", "variant_name": "Watermelon Slice"},
    )

    assert score < 1.0
    assert relation == "same_parent_different_variant"
    assert "llm_variant_name_conflict" in reasons


def test_run_metrics_tracks_openai_usage_and_cost() -> None:
    metrics = RunMetrics()
    metrics.add_usage(
        "gpt-5.4-nano",
        {
            "input_tokens": 1_000,
            "output_tokens": 200,
            "total_tokens": 1_200,
            "input_tokens_details": {"cached_tokens": 100},
        },
    )
    report = metrics.as_report(
        embedding_model="text-embedding-3-small",
        extraction_model="gpt-5.4-nano",
        input_records=10,
        total_elapsed_seconds=5.0,
    )

    usage = report["openai"]["usage_by_model"]["gpt-5.4-nano"]
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 1_000
    assert usage["cached_input_tokens"] == 100
    assert usage["output_tokens"] == 200
    assert usage["estimated_cost_usd"] > 0
    assert report["timing"]["avg_seconds_per_input_record"] == 0.5
