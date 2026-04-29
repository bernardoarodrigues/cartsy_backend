from __future__ import annotations

from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.scoring import score_pair


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


def test_exact_duplicate_scores_auto_merge() -> None:
    left = product(id="1")
    right = product(id="2", retailer="beleza_na_web", price="9199")
    result = score_pair(left, right, auto_threshold=0.86, review_threshold=0.70)
    assert result.decision == "auto_merge"
    assert result.score >= 0.86


def test_missing_size_does_not_reject_pair() -> None:
    left = product(id="1", dimension="473ml")
    right = product(id="2", prod_name="Cetaphil Loção Hidratante", dimension="")
    result = score_pair(left, right, auto_threshold=0.86, review_threshold=0.70)
    assert result.score >= 0.70
    assert "clearly_incompatible_size" not in result.explanation


def test_equivalent_size_boosts_confidence() -> None:
    left = product(id="1", prod_name="Cetaphil Moisturizing Lotion 16 fl oz", dimension="16 fl oz")
    right = product(id="2", prod_name="Cetaphil Loção Hidratante 473ml", dimension="473ml")
    result = score_pair(left, right, auto_threshold=0.86, review_threshold=0.70)
    assert "size_match" in result.explanation


def test_conflicting_clear_sizes_block_auto_merge() -> None:
    left = product(id="1", dimension="200ml", prod_name="Cetaphil Loção Hidratante 200ml")
    right = product(id="2", dimension="473ml", prod_name="Cetaphil Loção Hidratante 473ml")
    result = score_pair(left, right, auto_threshold=0.86, review_threshold=0.70)
    assert result.decision != "auto_merge"
    assert "clearly_incompatible_size" in result.explanation


def test_conflicting_brands_are_rejected_or_reviewed_not_merged() -> None:
    left = product(id="1", brand="Cetaphil")
    right = product(id="2", brand="CeraVe")
    result = score_pair(left, right, auto_threshold=0.86, review_threshold=0.70)
    assert result.decision != "auto_merge"
    assert "conflicting_strong_brand" in result.explanation
