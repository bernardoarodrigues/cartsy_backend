from __future__ import annotations

from cartsy_dedupe.pipeline import (
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
