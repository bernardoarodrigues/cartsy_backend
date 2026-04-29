from __future__ import annotations

from cartsy_dedupe.attributes import extract_model_tokens, extract_pack_count, extract_size


def test_extract_size_normalizes_liters_and_ounces() -> None:
    assert extract_size("Shampoo 1L")[0:2] == (1000.0, "ml")
    value, unit, ambiguous = extract_size("Tumbler 16 fl oz")
    assert round(value or 0) == 473
    assert unit == "ml"
    assert ambiguous is False


def test_extract_size_marks_multiple_sizes_ambiguous() -> None:
    value, unit, ambiguous = extract_size("Kit 2 unidades 100ml + tester 30ml")
    assert value == 100
    assert unit == "ml"
    assert ambiguous is True


def test_extract_pack_count() -> None:
    assert extract_pack_count("Kit com 2 unidades 100ml") == 2


def test_model_extraction_ignores_duration_tokens() -> None:
    tokens = extract_model_tokens("Base líquida duração 24h D3 claro")
    assert "24h" not in tokens
