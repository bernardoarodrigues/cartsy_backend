from __future__ import annotations

from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.scoring import MatchCertainty, RuleDecision, evaluate_rule


def product(**overrides: str):
    row = {
        "id": "1",
        "prod_name": "Cetaphil Loção Hidratante 473ml",
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
        "sku": "",
        "dimension": "473ml",
    }
    row.update(overrides)
    return normalize_row(row)


# ── CERTAIN_BLOCK ──────────────────────────────────────────────────────────────


def test_conflicting_global_id_blocks() -> None:
    left = product(id="1", specs='{"ean": "1234567890123"}')
    right = product(id="2", specs='{"ean": "9876543210987"}')
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.CERTAIN_BLOCK
    assert "global_id" in decision.reason


def test_conflicting_brand_blocks() -> None:
    left = product(id="1", brand="Cetaphil")
    right = product(id="2", brand="CeraVe")
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.CERTAIN_BLOCK
    assert "brand" in decision.reason


def test_conflicting_unambiguous_size_blocks() -> None:
    left = product(id="1", prod_name="Cetaphil Loção Hidratante 200ml", dimension="200ml")
    right = product(id="2", prod_name="Cetaphil Loção Hidratante 473ml", dimension="473ml")
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.CERTAIN_BLOCK
    assert "size" in decision.reason


def test_conflicting_pack_count_blocks() -> None:
    left = product(id="1", prod_name="Cetaphil Kit com 2 unidades", dimension="")
    right = product(id="2", prod_name="Cetaphil Kit com 3 unidades", dimension="")
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.CERTAIN_BLOCK
    assert "pack" in decision.reason


# ── CERTAIN_MATCH ──────────────────────────────────────────────────────────────


def test_ean_match_gives_certain_match() -> None:
    left = product(id="1", specs='{"ean": "1234567890123"}')
    right = product(id="2", specs='{"ean": "1234567890123"}', retailer="other_shop")
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.CERTAIN_MATCH
    assert "global_id" in decision.reason


def test_asin_match_gives_certain_match() -> None:
    left = product(id="1", sku="B08N5WRWNW")
    right = product(id="2", sku="B08N5WRWNW", retailer="other_shop", price="9199")
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.CERTAIN_MATCH
    assert "asin" in decision.reason


# ── STRONG_MATCH ───────────────────────────────────────────────────────────────


def test_same_retailer_sku_gives_strong_match() -> None:
    left = product(id="1", sku="SKU-001", dimension="")
    right = product(id="2", sku="SKU-001", price="9199", dimension="")
    decision = evaluate_rule(left, right)
    assert decision.certainty == MatchCertainty.STRONG_MATCH
    assert "retailer_sku" in decision.reason


def test_identical_title_no_model_tokens_gives_strong_match() -> None:
    left = product(id="1", prod_name="Cetaphil Loção Hidratante Corporal", dimension="")
    right = product(id="2", prod_name="Cetaphil Loção Hidratante Corporal", retailer="other_shop", dimension="")
    decision = evaluate_rule(left, right)
    assert decision.certainty in {MatchCertainty.STRONG_MATCH, MatchCertainty.CERTAIN_MATCH}


# ── LIKELY_MATCH ───────────────────────────────────────────────────────────────


def test_brand_title85_same_size_gives_likely_match() -> None:
    left = product(id="1", prod_name="Cetaphil Loção Hidratante Corporal 473ml")
    right = product(id="2", prod_name="Cetaphil Hidratante Corporal 473ml", retailer="other_shop")
    decision = evaluate_rule(left, right)
    assert decision.certainty in {MatchCertainty.LIKELY_MATCH, MatchCertainty.STRONG_MATCH}


# ── No false blocks ────────────────────────────────────────────────────────────


def test_missing_size_does_not_block() -> None:
    left = product(id="1", dimension="473ml")
    right = product(id="2", prod_name="Cetaphil Loção Hidratante", dimension="")
    decision = evaluate_rule(left, right)
    assert decision.certainty != MatchCertainty.CERTAIN_BLOCK


def test_ambiguous_size_does_not_block() -> None:
    # "473ml" on one side, same product but dimension field has two sizes → ambiguous.
    left = product(id="1", dimension="473ml")
    right = product(id="2", dimension="473ml 16fl oz")
    decision = evaluate_rule(left, right)
    assert decision.certainty != MatchCertainty.CERTAIN_BLOCK


def test_generic_brand_does_not_trigger_brand_block() -> None:
    # Two generic-brand products with different names but same size — brand mismatch should
    # not fire because "generic" is in GENERIC_BRANDS.
    left = product(id="1", brand="Generic", prod_name="Loção Hidratante 473ml")
    right = product(id="2", brand="Generic", prod_name="Hidratante Corporal 473ml")
    decision = evaluate_rule(left, right)
    assert decision.certainty != MatchCertainty.CERTAIN_BLOCK


# ── Return type ────────────────────────────────────────────────────────────────


def test_rule_decision_has_feature_scores() -> None:
    left = product(id="1")
    right = product(id="2", retailer="other_shop")
    decision = evaluate_rule(left, right)
    assert isinstance(decision, RuleDecision)
    assert isinstance(decision.feature_scores, dict)
    assert "brand" in decision.feature_scores
    assert "title" in decision.feature_scores
    assert "price" in decision.feature_scores
