from __future__ import annotations

import math
import re
from collections.abc import Mapping

from cartsy_dedupe.attributes import sizes_equivalent
from cartsy_dedupe.config import GLOBAL_IDENTIFIER_KEYS, MARKETPLACE_IDENTIFIER_KEYS
from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.text import STOPWORDS, normalize_text
from cartsy_dedupe.utils.pipeline_sql import postgres_retrieval_features

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency is declared for normal installs.
    import difflib

    class _FallbackFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FallbackFuzz()


DEFAULT_FEATURE_COLUMNS = [
    "same_retailer",
    "brand_exact",
    "brand_fuzzy",
    "title_token_set",
    "title_partial",
    "category_exact",
    "model_token_jaccard",
    "salient_token_jaccard",
    "size_match",
    "size_conflict",
    "pack_match",
    "pack_conflict",
    "price_ratio_diff",
    "price_both_present",
    "identifier_any",
    "exact_global_id",
    "exact_ean",
    "exact_gtin",
    "exact_upc",
    "exact_asin",
    "exact_retailer_sku",
    "exact_canonical_url",
    "exact_key_count",
    "exact_evidence_strength",
    "exact_sku_same_retailer",
    "exact_sku_cross_retailer",
    "rule_score",
    "rule_auto_blocked",
    "lexical_sim",
    "trigram_sim",
    "semantic_sim",
    "retrieval_layer_count",
    "variant_conflict",
]

GENERIC_TITLE_TOKENS = {
    "agua",
    "body",
    "capilar",
    "condicionador",
    "creme",
    "deodorant",
    "eau",
    "facial",
    "hair",
    "kit",
    "locao",
    "lotion",
    "mascara",
    "oil",
    "oleo",
    "parfum",
    "pecas",
    "perfume",
    "produto",
    "produtos",
    "serum",
    "shampoo",
    "spray",
    "toilette",
    "unidade",
    "unidades",
    "vodka",
}


def build_pair_features(
    left: NormalizedProduct,
    right: NormalizedProduct,
    block_keys: set[str],
    *,
    semantic_sim: float = 0.0,
    rule_score: float = 0.0,
    rule_auto_blocked: bool = False,
) -> dict[str, float]:
    """Build the stable pairwise ML features ported from the experiment.

    The feature names are intentionally kept stable because trained logistic
    regression bundles persist them as the model contract.
    """
    shared_identifier_keys = {
        key for key in left.identifiers if left.identifiers.get(key) and left.identifiers.get(key) == right.identifiers.get(key)
    }
    exact = exact_evidence_flags(block_keys)
    retrieval = postgres_retrieval_features(block_keys)
    price_ratio_diff = safe_ratio_diff(left.price_cents, right.price_cents)
    size_match, size_conflict = size_flags(left, right)
    pack_match = float(left.pack_count is not None and right.pack_count is not None and left.pack_count == right.pack_count)
    pack_conflict = float(left.pack_count is not None and right.pack_count is not None and left.pack_count != right.pack_count)
    left_salient = salient_title_tokens(left)
    right_salient = salient_title_tokens(right)

    features = {
        "same_retailer": float(left.retailer == right.retailer),
        "brand_exact": float(bool(left.brand_norm) and left.brand_norm == right.brand_norm),
        "brand_fuzzy": ratio(left.brand_norm, right.brand_norm),
        "title_token_set": token_set_ratio(left.name_norm, right.name_norm),
        "title_partial": partial_ratio(left.name_norm, right.name_norm),
        "category_exact": float(bool(left.category_leaf) and left.category_leaf == right.category_leaf),
        "model_token_jaccard": jaccard_score(set(left.model_tokens), set(right.model_tokens), default=0.4),
        "salient_token_jaccard": jaccard_score(left_salient, right_salient, default=0.45),
        "size_match": size_match,
        "size_conflict": size_conflict,
        "pack_match": pack_match,
        "pack_conflict": pack_conflict,
        "price_ratio_diff": 1.0 if math.isnan(price_ratio_diff) else price_ratio_diff,
        "price_both_present": float(not math.isnan(price_ratio_diff)),
        "identifier_any": float(bool(shared_identifier_keys) or exact["exact_key_count"] > 0),
        "exact_global_id": exact["exact_global_id"],
        "exact_ean": float("ean" in shared_identifier_keys),
        "exact_gtin": float("gtin" in shared_identifier_keys),
        "exact_upc": float("upc" in shared_identifier_keys),
        "exact_asin": max(float("asin" in shared_identifier_keys), exact["exact_asin"]),
        "exact_retailer_sku": exact["exact_retailer_sku"],
        "exact_canonical_url": exact["exact_canonical_url"],
        "exact_key_count": exact["exact_key_count"],
        "exact_evidence_strength": exact["exact_evidence_strength"],
        "exact_sku_same_retailer": max(
            float("sku" in shared_identifier_keys and left.retailer == right.retailer),
            exact["exact_retailer_sku"],
        ),
        "exact_sku_cross_retailer": float("sku" in shared_identifier_keys and left.retailer != right.retailer),
        "rule_score": clamp01(rule_score),
        "rule_auto_blocked": float(rule_auto_blocked),
        "lexical_sim": retrieval["lexical"],
        "trigram_sim": retrieval["trigram"],
        "semantic_sim": clamp01(semantic_sim),
        "retrieval_layer_count": retrieval_layer_count(retrieval, bool(shared_identifier_keys), semantic_sim),
        "variant_conflict": variant_conflict(left, right, left_salient=left_salient, right_salient=right_salient),
    }
    return {column: float(features[column]) for column in DEFAULT_FEATURE_COLUMNS}


def exact_evidence_flags(block_keys: set[str]) -> dict[str, float]:
    """Summarize exact retrieval evidence for both ML features and merge policy."""
    exact_types: set[str] = set()
    for block_key in block_keys:
        if not block_key.startswith("exact:"):
            continue
        parts = block_key.split(":", 2)
        if len(parts) < 3:
            continue
        exact_types.add(parts[1])

    exact_global = any(key in exact_types for key in GLOBAL_IDENTIFIER_KEYS)
    exact_asin = any(key in exact_types for key in MARKETPLACE_IDENTIFIER_KEYS)
    exact_retailer_sku = any(key.startswith("retailer_sku") for key in exact_types)
    exact_canonical_url = "canonical_url" in exact_types
    exact_key_count = float(len(exact_types))
    strength = 0.0
    if exact_global:
        strength = 1.0
    elif exact_asin:
        strength = 0.92
    elif exact_retailer_sku:
        strength = 0.88
    elif exact_canonical_url:
        strength = 0.86

    return {
        "exact_global_id": float(exact_global),
        "exact_asin": float(exact_asin),
        "exact_retailer_sku": float(exact_retailer_sku),
        "exact_canonical_url": float(exact_canonical_url),
        "exact_key_count": exact_key_count,
        "exact_evidence_strength": strength,
    }


def strong_exact_merge_reason(features: Mapping[str, float]) -> str:
    if float(features.get("exact_global_id", 0.0)) > 0.0:
        return "strong_exact:global_identifier"
    if float(features.get("exact_asin", 0.0)) > 0.0:
        return "strong_exact:asin"
    if float(features.get("exact_retailer_sku", 0.0)) > 0.0:
        return "strong_exact:retailer_sku"
    if float(features.get("exact_canonical_url", 0.0)) > 0.0:
        return "strong_exact:canonical_url"
    return ""


def feature_vector(features: Mapping[str, float], columns: list[str] | tuple[str, ...] = DEFAULT_FEATURE_COLUMNS) -> list[float]:
    return [float(features.get(column, 0.0)) for column in columns]


def hard_contradiction_features(features: Mapping[str, float]) -> bool:
    return any(
        float(features.get(column, 0.0)) >= 1.0
        for column in ("size_conflict", "pack_conflict", "variant_conflict")
    )


def salient_title_tokens(product: NormalizedProduct) -> set[str]:
    brand_tokens = set(normalize_text(product.brand_raw or product.brand_norm).split())
    category_tokens = set(normalize_text(product.category_leaf or product.category_norm).split())
    tokens: set[str] = set()
    for token in normalize_text(product.name_raw or product.name_norm).split():
        if len(token) <= 2:
            continue
        if token in STOPWORDS or token in GENERIC_TITLE_TOKENS or token in brand_tokens or token in category_tokens:
            continue
        if re.search(r"\d", token):
            continue
        tokens.add(token)
    return tokens


def jaccard_score(left_values: set[str], right_values: set[str], *, default: float = 0.0) -> float:
    if not left_values and not right_values:
        return default
    if not left_values or not right_values:
        return 0.0
    return len(left_values & right_values) / len(left_values | right_values)


def safe_ratio_diff(left_value: int | float | None, right_value: int | float | None) -> float:
    if left_value is None or right_value is None:
        return math.nan
    left_float = float(left_value)
    right_float = float(right_value)
    if left_float <= 0 or right_float <= 0:
        return math.nan
    return abs(left_float - right_float) / max(left_float, right_float)


def size_flags(left: NormalizedProduct, right: NormalizedProduct) -> tuple[float, float]:
    if left.size_value is None or right.size_value is None:
        return 0.0, 0.0
    if left.size_unit != right.size_unit:
        return 0.0, 0.0
    equivalent = sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit)
    return float(equivalent), float(not equivalent)


def variant_conflict(
    left: NormalizedProduct,
    right: NormalizedProduct,
    *,
    left_salient: set[str] | None = None,
    right_salient: set[str] | None = None,
) -> float:
    if left.brand_norm != right.brand_norm:
        return 0.0
    left_salient = salient_title_tokens(left) if left_salient is None else left_salient
    right_salient = salient_title_tokens(right) if right_salient is None else right_salient
    if not left_salient or not right_salient:
        return 0.0
    return float(not (left_salient & right_salient))


def retrieval_layer_count(retrieval: Mapping[str, float], has_identifier: bool, semantic_sim: float) -> float:
    return float(
        int(has_identifier)
        + int(float(retrieval.get("lexical", 0.0)) > 0.0)
        + int(float(retrieval.get("trigram", 0.0)) > 0.0)
        + int(float(semantic_sim) > 0.0)
    )


def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return clamp01(float(fuzz.ratio(left, right)) / 100.0)


def token_set_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return clamp01(float(fuzz.token_set_ratio(left, right)) / 100.0)


def partial_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return clamp01(float(fuzz.partial_ratio(left, right)) / 100.0)


def clamp01(value: float) -> float:
    if math.isnan(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
