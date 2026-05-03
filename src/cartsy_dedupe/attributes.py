from __future__ import annotations

import math
import re

from cartsy_dedupe.text import normalize_text

IDENTIFIER_PATTERNS = {
    "asin": re.compile(r"\bB0[A-Z0-9]{8}\b|\bB00[A-Z0-9]{7}\b", re.I),
    "ean": re.compile(r"\b\d{13}\b"),
    "upc": re.compile(r"\b\d{12}\b"),
}

SPEC_IDENTIFIER_KEYS = {
    "asin": "asin",
    "ean": "ean",
    "gtin": "gtin",
    "upc": "upc",
    "codigo ean": "ean",
    "código ean": "ean",
}

SIZE_RE = re.compile(
    r"(?<![a-z0-9])(\d+(?:[,.]\d+)?)\s*(fl\s*oz|floz|ml|l|g|kg|oz)(?![a-z])",
    re.I,
)
PACK_RE = re.compile(
    r"\b(?:pack\s*(?:of)?|kit|combo|dupla|trio|com|cont[eé]m)?\s*(\d{1,3})\s*"
    r"(?:produtos|produto|unidades|unidade|pcs|pecas|pe[cç]as|itens|items|un|x)\b",
    re.I,
)
MODEL_RE = re.compile(r"\b(?=[a-z0-9-]*\d)(?:[a-z]{1,5}-?\d[a-z0-9-]{2,}|\d+[a-z]{1,4}\d*)\b", re.I)


def parse_price_cents(value: str) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def extract_identifiers(
    source_sku: str,
    url: str,
    specs_obj: object | None,
    raw_text: str,
) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    sku = (source_sku or "").strip()
    if sku:
        identifiers["sku"] = normalize_text(sku)

    if isinstance(specs_obj, dict):
        for key, value in specs_obj.items():
            normalized_key = normalize_text(key)
            target_key = SPEC_IDENTIFIER_KEYS.get(normalized_key)
            if target_key and value:
                identifiers[target_key] = normalize_text(str(value))

    haystack = f"{source_sku} {url} {raw_text}"
    for key, pattern in IDENTIFIER_PATTERNS.items():
        if key not in identifiers:
            found = pattern.search(haystack or "")
            if found:
                identifiers[key] = normalize_text(found.group(0))
    return identifiers


def extract_model_tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    tokens: set[str] = set()
    for match in MODEL_RE.findall(normalized):
        token = normalize_text(match).replace(" ", "")
        if token and not _looks_like_size_only(token):
            tokens.add(token)
    return tuple(sorted(tokens))


def _looks_like_size_only(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(ml|g|kg|oz|l|h)?", token))


def extract_pack_count(text: str) -> int | None:
    normalized = normalize_text(text)
    match = PACK_RE.search(normalized)
    if not match:
        return None
    count = int(match.group(1))
    if count <= 1:
        return None
    return count


def extract_size(text: str) -> tuple[float | None, str | None, bool]:
    matches = list(SIZE_RE.finditer(text or ""))
    if not matches:
        return None, None, False

    converted: list[tuple[float, str]] = []
    for match in matches:
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2).lower().replace(" ", "")
        converted_value, converted_unit = convert_size(value, unit)
        converted.append((converted_value, converted_unit))

    first_value, first_unit = converted[0]
    unique = {(round(value, 2), unit) for value, unit in converted}
    ambiguous = len(unique) > 1
    return first_value, first_unit, ambiguous


def convert_size(value: float, unit: str) -> tuple[float, str]:
    if unit == "l":
        return value * 1000.0, "ml"
    if unit == "floz":
        return value * 29.5735, "ml"
    if unit == "fl oz":
        return value * 29.5735, "ml"
    if unit == "oz":
        return value * 28.3495, "g"
    if unit == "kg":
        return value * 1000.0, "g"
    return value, unit


def sizes_equivalent(a_value: float, a_unit: str, b_value: float, b_unit: str) -> bool:
    if a_unit != b_unit:
        return False
    if a_value == 0 or b_value == 0:
        return False
    tolerance = 0.08 if a_unit in {"ml", "g"} else 0.05
    return math.isclose(a_value, b_value, rel_tol=tolerance, abs_tol=1.0)
