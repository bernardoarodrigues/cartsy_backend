from __future__ import annotations

from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.text import flatten_jsonish, normalize_brand, normalize_category, normalize_text, parse_jsonish

from .attributes import (
    extract_identifiers,
    extract_model_tokens,
    extract_pack_count,
    extract_size,
    parse_price_cents,
)


def normalize_row(row: dict[str, str]) -> NormalizedProduct:
    name_raw = row.get("prod_name", "") or ""
    brand_raw = row.get("brand", "") or ""
    category_raw = row.get("category", "") or ""
    description_raw = row.get("description", "") or ""
    specs_raw = row.get("specs", "") or ""
    dimension_raw = row.get("dimension", "") or ""
    source_sku = row.get("sku", "") or ""
    url = row.get("url", "") or ""

    description_obj, bad_description_json = parse_jsonish(description_raw)
    specs_obj, bad_specs_json = parse_jsonish(specs_raw)
    description_norm = flatten_jsonish(description_obj) if description_obj is not None else normalize_text(description_raw)
    specs_text = flatten_jsonish(specs_obj) if specs_obj is not None else normalize_text(specs_raw)
    name_norm = normalize_text(name_raw)
    brand_norm = normalize_brand(brand_raw)
    category_norm, category_leaf = normalize_category(category_raw)
    price_cents = parse_price_cents(row.get("price", ""))

    attribute_text = " ".join(
        part
        for part in [name_raw, dimension_raw, specs_text, description_norm]
        if part
    )
    size_value, size_unit, size_ambiguous = extract_size(attribute_text)
    pack_count = extract_pack_count(attribute_text)
    model_tokens = extract_model_tokens(attribute_text)
    identifiers = extract_identifiers(source_sku, url, specs_obj, attribute_text)

    quality_flags = build_quality_flags(
        brand_norm=brand_norm,
        category_norm=category_norm,
        description_norm=description_norm,
        specs_text=specs_text,
        price_cents=price_cents,
        name_norm=name_norm,
        bad_description_json=bad_description_json,
        bad_specs_json=bad_specs_json,
        identifiers=identifiers,
    )

    return NormalizedProduct(
        source_id=str(row.get("id", "") or ""),
        retailer=row.get("retailer", "") or "",
        source_sku=source_sku,
        url=url,
        name_raw=name_raw,
        brand_raw=brand_raw,
        category_raw=category_raw,
        description_raw=description_raw,
        specs_raw=specs_raw,
        name_norm=name_norm,
        brand_norm=brand_norm,
        category_norm=category_norm,
        category_leaf=category_leaf,
        description_norm=description_norm,
        specs_text=specs_text,
        price_cents=price_cents,
        dimension_raw=dimension_raw,
        size_value=size_value,
        size_unit=size_unit,
        size_ambiguous=size_ambiguous,
        pack_count=pack_count,
        model_tokens=model_tokens,
        identifiers=identifiers,
        quality_flags=tuple(quality_flags),
    )


def build_quality_flags(
    *,
    brand_norm: str,
    category_norm: str,
    description_norm: str,
    specs_text: str,
    price_cents: int | None,
    name_norm: str,
    bad_description_json: bool,
    bad_specs_json: bool,
    identifiers: dict[str, str],
) -> list[str]:
    flags: list[str] = []
    if not brand_norm:
        flags.append("missing_brand")
    if brand_norm in {"generic", "generico", "genérico"}:
        flags.append("generic_brand")
    if not category_norm:
        flags.append("missing_category")
    if not description_norm:
        flags.append("missing_description")
    if not specs_text:
        flags.append("missing_specs")
    if price_cents is None:
        flags.append("missing_price")
    elif price_cents <= 0:
        flags.append("suspicious_price")
    if len(name_norm.split()) <= 1:
        flags.append("title_too_short")
    if bad_description_json:
        flags.append("invalid_description_json")
    if bad_specs_json:
        flags.append("invalid_specs_json")
    if not any(key in identifiers for key in ("ean", "gtin", "upc", "asin", "sku")):
        flags.append("missing_identifier")
    return flags
