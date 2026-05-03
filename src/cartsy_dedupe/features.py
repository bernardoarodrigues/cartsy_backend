"""ML feature extraction for product deduplication candidate pairs.

``build_pair_features`` produces the stable pairwise feature vector consumed
by the logistic-regression scorer.  ``DEFAULT_FEATURE_COLUMNS`` is the model
contract: adding, removing, or reordering columns invalidates existing
``.joblib`` bundles and requires retraining.

Rule-derived indicator features (``rule_certain_match``, ``rule_strong_match``,
``rule_likely_match``, ``rule_certain_block``) replace the former single
``rule_score`` float, giving the model interpretable certainty signals from the
condition chain.  ``feature_coverage_count`` measures how many independent
evidence channels carry non-zero signal for a pair, helping the model express
lower confidence on sparse-evidence decisions.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from cartsy_dedupe.attributes import sizes_equivalent
from cartsy_dedupe.config import GLOBAL_IDENTIFIER_KEYS, MARKETPLACE_IDENTIFIER_KEYS
from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.scoring import MatchCertainty, RuleDecision
from cartsy_dedupe.text import STOPWORDS, fuzz, normalize_text
from cartsy_dedupe.utils.pipeline_sql import postgres_retrieval_features

KIT_MARKERS = {
    "kit",
    "combo",
    "conjunto",
    "duo",
    "dupla",
    "trio",
}

KIT_COMPONENT_TERMS = {
    "ativador",
    "balm",
    "condicionador",
    "creme",
    "finalizador",
    "fluido",
    "gel",
    "geleia",
    "leave",
    "leite",
    "mascara",
    "mask",
    "oleo",
    "oil",
    "pomada",
    "protetor",
    "refil",
    "serum",
    "shampoo",
    "spray",
    "tonico",
}

PRODUCT_FORM_TERMS = {
    "ampola",
    "batom",
    "blush",
    "condicionador",
    "corretivo",
    "creme",
    "deo",
    "desodorante",
    "edp",
    "edt",
    "gel",
    "gloss",
    "hidratante",
    "leave",
    "locao",
    "mascara",
    "mask",
    "oleo",
    "parfum",
    "perfume",
    "po",
    "refil",
    "sabonete",
    "serum",
    "shampoo",
    "spray",
}

COLOR_VARIANT_TERMS = {
    "amarelo",
    "azul",
    "bege",
    "black",
    "blonde",
    "blue",
    "bronze",
    "brown",
    "cobre",
    "dourado",
    "gold",
    "grafite",
    "gray",
    "green",
    "grey",
    "ivory",
    "lilac",
    "loiro",
    "marrom",
    "nude",
    "pink",
    "preto",
    "purple",
    "red",
    "rosa",
    "roxo",
    "silver",
    "verde",
    "vermelho",
    "white",
}

VARIANT_WORD_PREFIXES = (
    "cor",
    "shade",
    "tom",
    "tone",
    "tono",
)

SHADE_CODE_RE = re.compile(r"\b(?:[a-z]{1,3}\d{1,3}[a-z]?|\d{1,3}[a-z]{1,3})\b", re.I)

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
    "rule_certain_match",
    "rule_strong_match",
    "rule_likely_match",
    "rule_certain_block",
    "lexical_sim",
    "trigram_sim",
    "semantic_sim",
    "retrieval_layer_count",
    "variant_conflict",
    "variant_token_conflict",
    "variant_token_presence_mismatch",
    "kit_standalone_conflict",
    "kit_count_conflict",
    "kit_component_conflict",
    "product_form_conflict",
    "weak_exact_contradiction",
    "contradiction_count",
    "contradiction_strength",
    "feature_coverage_count",
]

# Features that carry independent evidence; used to count active signals per pair.
_COVERAGE_INDICATORS: tuple[str, ...] = (
    "brand_exact",
    "exact_global_id",
    "exact_asin",
    "exact_retailer_sku",
    "exact_canonical_url",
    "size_match",
    "pack_match",
    "model_token_jaccard",
    "salient_token_jaccard",
    "rule_certain_match",
    "rule_strong_match",
    "rule_likely_match",
)


def build_pair_features(
    left: NormalizedProduct,
    right: NormalizedProduct,
    block_keys: set[str],
    *,
    semantic_sim: float = 0.0,
    rule_decision: RuleDecision | None = None,
) -> dict[str, float]:
    """Build the pairwise ML feature vector for a candidate product pair.

    Parameters
    ----------
    left, right:
        Normalized products to compare.
    block_keys:
        Retrieval evidence keys used to derive lexical/trigram/vector
        similarity features and exact-evidence flags.
    semantic_sim:
        Cosine similarity of the pair's dense embeddings.  ``0.0`` when
        embeddings are unavailable.
    rule_decision:
        Output of ``evaluate_rule``.  When ``None``, all rule indicator
        features are set to ``0.0``.

    Returns
    -------
    dict mapping each column in ``DEFAULT_FEATURE_COLUMNS`` to a float.
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
    contradiction = contradiction_features(
        left,
        right,
        left_salient=left_salient,
        right_salient=right_salient,
    )

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
        "rule_certain_match": float(rule_decision is not None and rule_decision.certainty == MatchCertainty.CERTAIN_MATCH),
        "rule_strong_match":  float(rule_decision is not None and rule_decision.certainty == MatchCertainty.STRONG_MATCH),
        "rule_likely_match":  float(rule_decision is not None and rule_decision.certainty == MatchCertainty.LIKELY_MATCH),
        "rule_certain_block": float(rule_decision is not None and rule_decision.certainty == MatchCertainty.CERTAIN_BLOCK),
        "lexical_sim": retrieval["lexical"],
        "trigram_sim": retrieval["trigram"],
        "semantic_sim": clamp01(semantic_sim),
        "retrieval_layer_count": retrieval_layer_count(retrieval, bool(shared_identifier_keys), semantic_sim),
        **contradiction,
    }
    features["feature_coverage_count"] = float(
        sum(1 for col in _COVERAGE_INDICATORS if features.get(col, 0.0) > 0.0)
    )
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


def feature_vector(features: Mapping[str, float], columns: list[str] | tuple[str, ...] = DEFAULT_FEATURE_COLUMNS) -> list[float]:
    """Return a dense float list aligned to ``columns``."""
    return [float(features.get(column, 0.0)) for column in columns]


def hard_contradiction_features(features: Mapping[str, float]) -> bool:
    """Return True if high-confidence identity contradictions are present."""
    return any(
        float(features.get(column, 0.0)) >= 1.0
        for column in (
            "size_conflict",
            "pack_conflict",
            "variant_token_conflict",
            "variant_token_presence_mismatch",
            "kit_standalone_conflict",
            "kit_count_conflict",
            "kit_component_conflict",
        )
    )


def salient_title_tokens(product: NormalizedProduct) -> set[str]:
    """Extract title tokens that carry discriminative signal for a product.

    Filters out stop-words, brand tokens, and category tokens.  Skips tokens
    that contain digits, which are better handled by dedicated size and
    model-token features.
    """
    brand_tokens = set(normalize_text(product.brand_raw or product.brand_norm).split())
    category_tokens = set(normalize_text(product.category_leaf or product.category_norm).split())
    tokens: set[str] = set()
    for token in normalize_text(product.name_raw or product.name_norm).split():
        if len(token) <= 2:
            continue
        if token in STOPWORDS or token in brand_tokens or token in category_tokens:
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


def contradiction_features(
    left: NormalizedProduct,
    right: NormalizedProduct,
    *,
    left_salient: set[str] | None = None,
    right_salient: set[str] | None = None,
) -> dict[str, float]:
    """Return reusable product-identity contradiction features.

    These features describe general product identity concepts rather than
    brand-specific edge cases, so the logistic-regression trainer can learn
    negative evidence weights from shade, kit, form, and weak-exact conflicts.
    The runtime policy only hard-blocks the highest-confidence subset.
    """
    left_salient = salient_title_tokens(left) if left_salient is None else left_salient
    right_salient = salient_title_tokens(right) if right_salient is None else right_salient
    same_brand = bool(left.brand_norm) and left.brand_norm == right.brand_norm

    left_tokens = product_identity_tokens(left)
    right_tokens = product_identity_tokens(right)
    left_variant = variant_tokens(left, left_tokens)
    right_variant = variant_tokens(right, right_tokens)
    left_components = component_terms(left_tokens)
    right_components = component_terms(right_tokens)
    left_forms = form_terms(left_tokens)
    right_forms = form_terms(right_tokens)
    left_is_kit = is_kit_product(left, left_tokens, left_components)
    right_is_kit = is_kit_product(right, right_tokens, right_components)

    broad_variant_conflict = variant_conflict(
        left,
        right,
        left_salient=left_salient,
        right_salient=right_salient,
    )
    variant_token_conflict = float(
        same_brand_or_shared_identifier(left, right)
        and bool(left_variant)
        and bool(right_variant)
        and left_variant.isdisjoint(right_variant)
    )
    variant_token_presence_mismatch = float(
        same_brand_or_shared_identifier(left, right)
        and bool(left_variant) != bool(right_variant)
        and high_title_containment(left.name_norm, right.name_norm)
    )
    kit_standalone_conflict = float(left_is_kit != right_is_kit and bool(left_components or right_components))
    kit_count_conflict = float(
        left_is_kit
        and right_is_kit
        and left.pack_count is not None
        and right.pack_count is not None
        and left.pack_count != right.pack_count
    )
    kit_component_conflict = float(
        left_is_kit
        and right_is_kit
        and bool(left_components)
        and bool(right_components)
        and left_components.isdisjoint(right_components)
    )
    product_form_conflict = float(
        not left_is_kit
        and not right_is_kit
        and bool(left_forms)
        and bool(right_forms)
        and left_forms.isdisjoint(right_forms)
    )

    weak_exact_contradiction = float(
        exact_identifier_present(left, right)
        and (
            variant_token_conflict
            or variant_token_presence_mismatch
            or kit_standalone_conflict
            or kit_count_conflict
            or kit_component_conflict
            or (product_form_conflict and broad_variant_conflict)
        )
    )
    contradiction_values = (
        broad_variant_conflict,
        variant_token_conflict,
        variant_token_presence_mismatch,
        kit_standalone_conflict,
        kit_count_conflict,
        kit_component_conflict,
        product_form_conflict,
        weak_exact_contradiction,
    )
    contradiction_count = float(sum(1 for value in contradiction_values if value > 0.0))
    contradiction_strength = max(contradiction_values, default=0.0)

    return {
        "variant_conflict": broad_variant_conflict,
        "variant_token_conflict": variant_token_conflict,
        "variant_token_presence_mismatch": variant_token_presence_mismatch,
        "kit_standalone_conflict": kit_standalone_conflict,
        "kit_count_conflict": kit_count_conflict,
        "kit_component_conflict": kit_component_conflict,
        "product_form_conflict": product_form_conflict,
        "weak_exact_contradiction": weak_exact_contradiction,
        "contradiction_count": contradiction_count,
        "contradiction_strength": contradiction_strength,
    }


def product_identity_tokens(product: NormalizedProduct) -> set[str]:
    text = " ".join(
        part
        for part in (
            product.name_raw,
            product.name_norm,
            product.category_leaf,
            product.dimension_raw,
        )
        if part
    )
    return set(normalize_text(text).split())


def variant_tokens(product: NormalizedProduct, tokens: set[str]) -> set[str]:
    values = {token for token in tokens if token in COLOR_VARIANT_TERMS}
    normalized_title = normalize_text(product.name_raw or product.name_norm)
    for match in SHADE_CODE_RE.findall(normalized_title):
        token = normalize_text(match).replace(" ", "")
        if token and not _size_like_variant_token(token):
            values.add(token)
    split_tokens = normalized_title.split()
    for index, token in enumerate(split_tokens[:-1]):
        if token in VARIANT_WORD_PREFIXES:
            next_token = split_tokens[index + 1]
            if next_token not in STOPWORDS and not _size_like_variant_token(next_token):
                values.add(next_token)
    return values


def _size_like_variant_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:ml|g|kg|l|oz)?", token))


def component_terms(tokens: set[str]) -> set[str]:
    return {token for token in tokens if token in KIT_COMPONENT_TERMS}


def form_terms(tokens: set[str]) -> set[str]:
    return {token for token in tokens if token in PRODUCT_FORM_TERMS}


def is_kit_product(product: NormalizedProduct, tokens: set[str], components: set[str]) -> bool:
    if product.pack_count is not None and product.pack_count > 1:
        return True
    if tokens & KIT_MARKERS:
        return True
    title = product.name_raw or product.name_norm
    return "+" in title and len(components) >= 2


def exact_identifier_present(left: NormalizedProduct, right: NormalizedProduct) -> bool:
    return any(
        left.identifiers.get(key) and left.identifiers.get(key) == right.identifiers.get(key)
        for key in ("ean", "gtin", "upc", "asin", "sku")
    )


def same_brand_or_shared_identifier(left: NormalizedProduct, right: NormalizedProduct) -> bool:
    return (bool(left.brand_norm) and left.brand_norm == right.brand_norm) or exact_identifier_present(left, right)


def high_title_containment(left_title: str, right_title: str) -> bool:
    if not left_title or not right_title:
        return False
    left_tokens = set(left_title.split())
    right_tokens = set(right_title.split())
    if not left_tokens or not right_tokens:
        return False
    smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    return len(smaller & larger) / len(smaller) >= 0.80


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
