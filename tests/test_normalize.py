from __future__ import annotations

from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.text import normalize_brand, normalize_category, normalize_text


def test_text_normalization_removes_accents_and_punctuation() -> None:
    assert normalize_text("Óleo Wella - 30ml!") == "oleo wella 30ml"


def test_brand_normalization_removes_symbols_spaces_and_accents() -> None:
    assert normalize_brand("L'Oréal Paris") == "lorealparis"
    assert normalize_brand("L OREAL-PARIS") == "lorealparis"
    category, leaf = normalize_category("Beleza›Pele›Rosto›Hidratantes")
    assert category == "beleza>pele>rosto>hidratantes"
    assert leaf == "hidratantes"


def test_normalize_row_extracts_json_identifiers_and_quality_flags() -> None:
    product = normalize_row(
        {
            "id": "1",
            "prod_name": "CeraVe Loção Hidratante 473ml",
            "brand": "CeraVe",
            "category": "Beleza>Pele",
            "description": '["hidratante corporal"]',
            "specs": '{"EAN": "1234567890123", "ASIN": "B07C5XYT19"}',
            "img_links": "",
            "url": "https://example.com/p",
            "created_at": "",
            "updated_at": "",
            "retailer": "amazon_br",
            "price": "6790",
            "sku": "B07C5XYT19",
            "dimension": "473ml",
        }
    )
    assert product.size_value == 473
    assert product.size_unit == "ml"
    assert product.identifiers["ean"] == "1234567890123"
    assert product.identifiers["asin"] == "b07c5xyt19"
    assert "missing_description" not in product.quality_flags


def test_invalid_json_is_flagged_without_crashing() -> None:
    product = normalize_row(
        {
            "id": "2",
            "prod_name": "Produto Teste",
            "brand": "",
            "category": "",
            "description": "[bad",
            "specs": "{bad",
            "img_links": "",
            "url": "",
            "created_at": "",
            "updated_at": "",
            "retailer": "x",
            "price": "",
            "sku": "",
            "dimension": "",
        }
    )
    assert "invalid_description_json" in product.quality_flags
    assert "invalid_specs_json" in product.quality_flags
    assert "missing_brand" in product.quality_flags


def test_normalization_does_not_guess_open_ended_variant_terms() -> None:
    product = normalize_row(
        {
            "id": "3",
            "prod_name": "Base líquida cor L60 baunilha",
            "brand": "Marca",
            "category": "Beleza>Maquiagem",
            "description": "",
            "specs": "{}",
            "img_links": "",
            "url": "",
            "created_at": "",
            "updated_at": "",
            "retailer": "x",
            "price": "1000",
            "sku": "",
            "dimension": "",
        }
    )

    assert not hasattr(product, "color")
    assert not hasattr(product, "shade")
    assert not hasattr(product, "scent")
    assert product.extracted_attributes == {}
