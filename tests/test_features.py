from __future__ import annotations

from cartsy_dedupe.features import DEFAULT_FEATURE_COLUMNS, build_pair_features, hard_contradiction_features
from cartsy_dedupe.normalize import normalize_row


def product(**overrides: str):
    row = {
        "id": "1",
        "prod_name": "Cetaphil Locao Hidratante 473ml",
        "brand": "Cetaphil",
        "category": "Beleza>Pele>Hidratantes",
        "description": '["hidratante corporal"]',
        "specs": "{}",
        "img_links": "",
        "url": "",
        "created_at": "",
        "updated_at": "",
        "retailer": "amazon_br",
        "price": "6790",
        "sku": "SKU-1",
        "dimension": "473ml",
    }
    row.update(overrides)
    return normalize_row(row)


def test_pair_features_match_experiment_contract() -> None:
    left = product(id="1")
    right = product(id="2", retailer="other_shop", price="6990", sku="SKU-2")

    features = build_pair_features(
        left,
        right,
        {"lexical:fts:0.5000", "trigram:title:0.9000", "vector:cosine:0.8800"},
        semantic_sim=0.93,
    )

    assert list(features) == DEFAULT_FEATURE_COLUMNS
    assert features["brand_exact"] == 1.0
    assert features["title_token_set"] >= 0.95
    assert features["lexical_sim"] == 0.7
    assert features["trigram_sim"] == 0.9
    assert features["semantic_sim"] == 0.93
    assert features["retrieval_layer_count"] == 3.0
    assert features["price_both_present"] == 1.0
    assert "exact_global_id" in DEFAULT_FEATURE_COLUMNS
    assert "rule_certain_match" in DEFAULT_FEATURE_COLUMNS
    assert "feature_coverage_count" in DEFAULT_FEATURE_COLUMNS


def test_pair_features_include_identifier_and_variant_conflicts() -> None:
    left = product(id="1", prod_name="Cetaphil Batom Rosa 30ml", sku="SHARED")
    right = product(id="2", prod_name="Cetaphil Gloss Azul 50ml", sku="SHARED", dimension="50ml")

    features = build_pair_features(left, right, {"exact:retailer_sku:amazon_br:SHARED"}, semantic_sim=0.91)

    assert features["identifier_any"] == 1.0
    assert features["exact_retailer_sku"] == 1.0
    assert features["exact_key_count"] == 1.0
    assert features["exact_sku_same_retailer"] == 1.0
    assert features["size_conflict"] == 1.0
    assert features["variant_conflict"] == 1.0
    assert hard_contradiction_features(features)


def test_pair_features_expose_exact_canonical_url_evidence() -> None:
    left = product(id="1", url="https://example.com/products/cetaphil-473ml")
    right = product(id="2", url="https://example.com/products/cetaphil-473ml")

    features = build_pair_features(left, right, {"exact:canonical_url:example.com/products/cetaphil-473ml"})

    assert features["identifier_any"] == 1.0
    assert features["exact_canonical_url"] == 1.0
    assert features["exact_evidence_strength"] >= 0.86
