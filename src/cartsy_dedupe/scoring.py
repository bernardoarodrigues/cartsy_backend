"""Rule-based condition evaluator for product pair matching.

Evaluates an ordered condition chain and returns a ``RuleDecision`` with a
``MatchCertainty`` level.  Certainty levels range from ``CERTAIN_BLOCK``
(hard contradiction found) to ``CERTAIN_MATCH`` (exact global identifier
confirmed), with intermediate levels ``STRONG_MATCH``, ``LIKELY_MATCH``, and
``UNCERTAIN`` used as ML indicator features for borderline pair scoring.

The evaluator never performs arithmetic scoring — every branch maps to a named
condition so that the reason for each decision can be traced directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cartsy_dedupe.attributes import sizes_equivalent
from cartsy_dedupe.config import GENERIC_BRANDS, GLOBAL_IDENTIFIER_KEYS, MARKETPLACE_IDENTIFIER_KEYS
from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.utils.pipeline_helpers import canonicalize_url

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    import difflib

    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FallbackFuzz()


class MatchCertainty(str, Enum):
    """Certainty level assigned by the rule condition chain.

    ``CERTAIN_MATCH`` and ``CERTAIN_BLOCK`` bypass ML scoring entirely.
    All other levels are encoded as binary indicator features in the ML
    feature vector.
    """

    CERTAIN_BLOCK = "CERTAIN_BLOCK"  # hard contradiction — do not merge
    UNCERTAIN     = "UNCERTAIN"      # no clear signal — defer to ML
    LIKELY_MATCH  = "LIKELY_MATCH"   # brand + title + corroborating signal
    STRONG_MATCH  = "STRONG_MATCH"   # retailer SKU or brand + very high title sim
    CERTAIN_MATCH = "CERTAIN_MATCH"  # exact global identifier or canonical URL


@dataclass(frozen=True)
class RuleDecision:
    """Result of the rule condition chain for a candidate product pair."""

    certainty: MatchCertainty
    reason: str                       # name of the first-match condition
    feature_scores: dict[str, float]  # diagnostic component scores for observability


def evaluate_rule(left: NormalizedProduct, right: NormalizedProduct) -> RuleDecision:
    """Evaluate the ordered condition chain for a candidate product pair.

    Conditions are checked top-to-bottom; the first match determines the
    returned ``RuleDecision``.  ``CERTAIN_MATCH`` signals result in an
    immediate merge (score=1.0) without ML inference; ``CERTAIN_BLOCK``
    signals result in an immediate no-merge (score=0.0).  All other levels
    feed the ML model as binary indicator features.

    Parameters
    ----------
    left, right:
        Normalized products to compare.
    """
    feature_scores = _compute_component_scores(left, right)

    # ── CERTAIN_BLOCK ──────────────────────────────────────────────────────────
    # Each condition below represents an irreconcilable factual contradiction.

    # Same identifier key present on both sides with different values.
    for key in GLOBAL_IDENTIFIER_KEYS:
        lv = left.identifiers.get(key)
        rv = right.identifiers.get(key)
        if lv and rv and lv != rv:
            return RuleDecision(MatchCertainty.CERTAIN_BLOCK, f"conflicting_global_id:{key}", feature_scores)

    # Both brands are known, non-generic, and clearly different.
    if (
        left.brand_norm and right.brand_norm
        and left.brand_norm not in GENERIC_BRANDS
        and right.brand_norm not in GENERIC_BRANDS
        and left.brand_norm != right.brand_norm
        and string_similarity(left.brand_norm, right.brand_norm) < 0.88
    ):
        return RuleDecision(MatchCertainty.CERTAIN_BLOCK, "conflicting_strong_brand", feature_scores)

    # Both products have a clear, unambiguous size in the same unit and they differ.
    if (
        left.size_value is not None and right.size_value is not None
        and not left.size_ambiguous and not right.size_ambiguous
        and left.size_unit == right.size_unit
        and not sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit)
    ):
        return RuleDecision(MatchCertainty.CERTAIN_BLOCK, "conflicting_unambiguous_size", feature_scores)

    # Both products have an explicit pack count and it differs.
    if left.pack_count is not None and right.pack_count is not None and left.pack_count != right.pack_count:
        return RuleDecision(MatchCertainty.CERTAIN_BLOCK, "conflicting_pack_count", feature_scores)

    # ── CERTAIN_MATCH ──────────────────────────────────────────────────────────

    global_id_matches = _matching_global_identifiers(left, right)

    # Any EAN, GTIN, or UPC agrees.
    for key in GLOBAL_IDENTIFIER_KEYS:
        if key in global_id_matches:
            return RuleDecision(MatchCertainty.CERTAIN_MATCH, f"exact_global_id:{key}", feature_scores)

    # ASIN agrees (marketplace identifier — strong but not universal).
    if "asin" in global_id_matches:
        return RuleDecision(MatchCertainty.CERTAIN_MATCH, "exact_asin", feature_scores)

    # Canonical product-page URL agrees after normalization.
    left_url = canonicalize_url(left.url)
    right_url = canonicalize_url(right.url)
    if left_url and right_url and left_url == right_url:
        return RuleDecision(MatchCertainty.CERTAIN_MATCH, "exact_canonical_url", feature_scores)

    # ── STRONG_MATCH ───────────────────────────────────────────────────────────

    brands_exact = bool(left.brand_norm) and left.brand_norm == right.brand_norm
    title_sim = string_similarity(left.name_norm, right.name_norm)
    model_overlap = bool(set(left.model_tokens) & set(right.model_tokens))
    models_absent = not left.model_tokens or not right.model_tokens

    # Same retailer + same SKU: the source itself considers them the same item.
    if (
        left.retailer and left.retailer == right.retailer
        and left.identifiers.get("sku")
        and left.identifiers.get("sku") == right.identifiers.get("sku")
    ):
        return RuleDecision(MatchCertainty.STRONG_MATCH, "same_retailer_sku", feature_scores)

    # Brand exact + near-identical title + confirmed model token overlap.
    if brands_exact and title_sim >= 0.95 and not models_absent and model_overlap:
        return RuleDecision(MatchCertainty.STRONG_MATCH, "brand_title95_model_overlap", feature_scores)

    # Brand exact + very high title similarity when model tokens are unavailable.
    if brands_exact and title_sim >= 0.97 and models_absent:
        return RuleDecision(MatchCertainty.STRONG_MATCH, "brand_title97_no_model", feature_scores)

    # ── LIKELY_MATCH ───────────────────────────────────────────────────────────

    size_matches = (
        left.size_value is not None and right.size_value is not None
        and left.size_unit == right.size_unit
        and sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit)
    )
    no_pack_conflict = not (
        left.pack_count is not None
        and right.pack_count is not None
        and left.pack_count != right.pack_count
    )

    # Brand exact + high title similarity + size corroborates.
    if brands_exact and title_sim >= 0.85 and size_matches and no_pack_conflict:
        return RuleDecision(MatchCertainty.LIKELY_MATCH, "brand_title85_size_match", feature_scores)

    # Brand exact + model token overlap + decent title similarity.
    if brands_exact and model_overlap and title_sim >= 0.70:
        return RuleDecision(MatchCertainty.LIKELY_MATCH, "brand_model_overlap_title70", feature_scores)

    # ── UNCERTAIN ──────────────────────────────────────────────────────────────

    return RuleDecision(MatchCertainty.UNCERTAIN, "uncertain", feature_scores)


def string_similarity(left: str, right: str) -> float:
    """Return token-set similarity in [0, 1] between two strings.

    Returns ``0.0`` if either string is empty.  Uses ``rapidfuzz.fuzz.token_set_ratio``
    when available, falling back to ``difflib.SequenceMatcher``.
    """
    if not left or not right:
        return 0.0
    return float(fuzz.token_set_ratio(left, right)) / 100.0


# ── Private helpers ────────────────────────────────────────────────────────────


def _matching_global_identifiers(left: NormalizedProduct, right: NormalizedProduct) -> set[str]:
    matches: set[str] = set()
    for key in GLOBAL_IDENTIFIER_KEYS | MARKETPLACE_IDENTIFIER_KEYS:
        if left.identifiers.get(key) and left.identifiers.get(key) == right.identifiers.get(key):
            matches.add(key)
    return matches


def _compute_component_scores(left: NormalizedProduct, right: NormalizedProduct) -> dict[str, float]:
    """Compute diagnostic sub-scores for observability in CandidatePair.feature_scores."""
    return {
        "brand": _brand_component(left, right),
        "title": _title_component(left, right),
        "identifier": _identifier_component(left, right),
        "model": _model_component(left, right),
        "variant": _variant_component(left, right),
        "category": _category_component(left, right),
        "specs_description": _specs_description_component(left, right),
        "price": _price_component(left, right),
    }


def _brand_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    if not left.brand_norm or not right.brand_norm:
        return 0.35
    if left.brand_norm in GENERIC_BRANDS or right.brand_norm in GENERIC_BRANDS:
        return 0.30
    if left.brand_norm == right.brand_norm:
        return 1.0
    similarity = string_similarity(left.brand_norm, right.brand_norm)
    return 0.82 if similarity >= 0.88 else 0.0


def _title_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    return string_similarity(left.name_norm, right.name_norm)


def _identifier_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    matches = _matching_global_identifiers(left, right)
    if matches:
        return 1.0
    for key in GLOBAL_IDENTIFIER_KEYS:
        if left.identifiers.get(key) and right.identifiers.get(key) and left.identifiers[key] != right.identifiers[key]:
            return 0.0
    if (
        left.retailer and left.retailer == right.retailer
        and left.identifiers.get("sku")
        and left.identifiers.get("sku") == right.identifiers.get("sku")
    ):
        return 0.82
    if left.identifiers.get("sku") and right.identifiers.get("sku") and left.identifiers["sku"] == right.identifiers["sku"]:
        return 0.65
    return 0.20


def _model_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    if not left.model_tokens and not right.model_tokens:
        return 0.45
    if not left.model_tokens or not right.model_tokens:
        return 0.45
    overlap = set(left.model_tokens) & set(right.model_tokens)
    return 1.0 if overlap else 0.0


def _variant_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    scores: list[float] = [_size_component(left, right)]
    if left.pack_count and right.pack_count:
        scores.append(1.0 if left.pack_count == right.pack_count else 0.0)
    elif left.pack_count or right.pack_count:
        scores.append(0.55)
    return sum(scores) / len(scores)


def _size_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    if left.size_value is None and right.size_value is None:
        return 0.55
    if left.size_value is None or right.size_value is None:
        return 0.62
    if left.size_ambiguous or right.size_ambiguous:
        if left.size_unit == right.size_unit and sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit):
            return 0.78
        return 0.55
    if left.size_unit == right.size_unit and sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit):
        return 1.0
    return 0.0


def _category_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    if not left.category_norm or not right.category_norm:
        return 0.45
    if left.category_norm == right.category_norm:
        return 1.0
    if left.category_leaf and left.category_leaf == right.category_leaf:
        return 0.78
    return string_similarity(left.category_norm, right.category_norm) * 0.8


def _specs_description_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    left_text = " ".join(part for part in [left.specs_text, left.description_norm] if part)
    right_text = " ".join(part for part in [right.specs_text, right.description_norm] if part)
    if not left_text or not right_text:
        return 0.35
    return string_similarity(left_text[:1_500], right_text[:1_500])


def _price_component(left: NormalizedProduct, right: NormalizedProduct) -> float:
    if left.price_cents is None or right.price_cents is None:
        return 0.45
    high = max(left.price_cents, right.price_cents)
    low = min(left.price_cents, right.price_cents)
    if high <= 0:
        return 0.0
    ratio = low / high
    if ratio >= 0.85:
        return 1.0
    return 0.65 if ratio >= 0.55 else 0.25
