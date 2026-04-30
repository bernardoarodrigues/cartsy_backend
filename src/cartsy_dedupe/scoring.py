from __future__ import annotations

from dataclasses import dataclass

from .attributes import sizes_equivalent
from .config import GENERIC_BRANDS, GLOBAL_IDENTIFIER_KEYS, MARKETPLACE_IDENTIFIER_KEYS
from .schemas import NormalizedProduct

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


@dataclass(frozen=True)
class ScoreResult:
    score: float
    decision: str
    explanation: str
    feature_scores: dict[str, float]
    auto_blocked: bool


def score_pair(
    left: NormalizedProduct,
    right: NormalizedProduct,
    *,
    merge_threshold: float,
) -> ScoreResult:
    reasons: list[str] = []
    contradictions: list[str] = []
    feature_scores = {
        "brand": brand_score(left, right, reasons, contradictions),
        "title": title_score(left, right, reasons),
        "identifier": identifier_score(left, right, reasons, contradictions),
        "model": model_score(left, right, reasons, contradictions),
        "variant": variant_score(left, right, reasons, contradictions),
        "category": category_score(left, right, reasons),
        "specs_description": specs_description_score(left, right),
        "price": price_score(left, right, reasons),
    }

    score = (
        0.18 * feature_scores["brand"]
        + 0.24 * feature_scores["title"]
        + 0.18 * feature_scores["identifier"]
        + 0.15 * feature_scores["model"]
        + 0.13 * feature_scores["variant"]
        + 0.06 * feature_scores["category"]
        + 0.04 * feature_scores["specs_description"]
        + 0.02 * feature_scores["price"]
    )

    if any(key in matching_global_identifiers(left, right) for key in GLOBAL_IDENTIFIER_KEYS):
        score = max(score, 0.94)
    elif "asin" in matching_global_identifiers(left, right):
        score = max(score, 0.90)
    elif (
        feature_scores["brand"] >= 0.95
        and feature_scores["title"] >= 0.95
        and feature_scores["variant"] >= 0.85
    ):
        score = max(score, 0.88)

    hard_block = any(
        reason in contradictions
        for reason in (
            "conflicting_strong_brand",
            "conflicting_global_identifier",
            "conflicting_model",
            "clearly_incompatible_size",
            "clearly_incompatible_variant",
        )
    )
    if hard_block:
        score = min(score, 0.84)

    score = max(0.0, min(1.0, score))
    decision = "merge" if score >= merge_threshold and not hard_block else "no_merge"

    explanation_parts = reasons + [f"penalty:{item}" for item in contradictions]
    return ScoreResult(
        score=score,
        decision=decision,
        explanation="; ".join(explanation_parts[:12]),
        feature_scores=feature_scores,
        auto_blocked=hard_block,
    )


def string_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(fuzz.token_set_ratio(left, right)) / 100.0


def brand_score(
    left: NormalizedProduct,
    right: NormalizedProduct,
    reasons: list[str],
    contradictions: list[str],
) -> float:
    if not left.brand_norm or not right.brand_norm:
        reasons.append("brand_missing")
        return 0.35
    if left.brand_norm in GENERIC_BRANDS or right.brand_norm in GENERIC_BRANDS:
        reasons.append("generic_brand")
        return 0.30
    if left.brand_norm == right.brand_norm:
        reasons.append(f"brand_match:{left.brand_norm}")
        return 1.0
    similarity = string_similarity(left.brand_norm, right.brand_norm)
    if similarity >= 0.88:
        reasons.append(f"brand_near_match:{similarity:.2f}")
        return 0.82
    contradictions.append("conflicting_strong_brand")
    return 0.0


def title_score(left: NormalizedProduct, right: NormalizedProduct, reasons: list[str]) -> float:
    score = string_similarity(left.name_norm, right.name_norm)
    if score >= 0.90:
        reasons.append(f"title_high:{score:.2f}")
    elif score >= 0.75:
        reasons.append(f"title_medium:{score:.2f}")
    return score


def matching_global_identifiers(left: NormalizedProduct, right: NormalizedProduct) -> set[str]:
    matches: set[str] = set()
    for key in GLOBAL_IDENTIFIER_KEYS | MARKETPLACE_IDENTIFIER_KEYS:
        if left.identifiers.get(key) and left.identifiers.get(key) == right.identifiers.get(key):
            matches.add(key)
    return matches


def identifier_score(
    left: NormalizedProduct,
    right: NormalizedProduct,
    reasons: list[str],
    contradictions: list[str],
) -> float:
    matches = matching_global_identifiers(left, right)
    if matches:
        reasons.append("identifier_match:" + ",".join(sorted(matches)))
        return 1.0

    for key in GLOBAL_IDENTIFIER_KEYS:
        if left.identifiers.get(key) and right.identifiers.get(key) and left.identifiers[key] != right.identifiers[key]:
            contradictions.append("conflicting_global_identifier")
            return 0.0

    if (
        left.retailer
        and left.retailer == right.retailer
        and left.identifiers.get("sku")
        and left.identifiers.get("sku") == right.identifiers.get("sku")
    ):
        reasons.append("same_retailer_sku")
        return 0.82
    if left.identifiers.get("sku") and right.identifiers.get("sku") and left.identifiers["sku"] == right.identifiers["sku"]:
        reasons.append("sku_match_cross_retailer")
        return 0.65
    return 0.20


def model_score(
    left: NormalizedProduct,
    right: NormalizedProduct,
    reasons: list[str],
    contradictions: list[str],
) -> float:
    if not left.model_tokens and not right.model_tokens:
        return 0.45
    if not left.model_tokens or not right.model_tokens:
        reasons.append("model_missing_one_side")
        return 0.45
    overlap = set(left.model_tokens) & set(right.model_tokens)
    if overlap:
        reasons.append("model_match:" + ",".join(sorted(overlap)[:3]))
        return 1.0
    if left.brand_norm == right.brand_norm and left.brand_norm not in GENERIC_BRANDS:
        contradictions.append("conflicting_model")
    return 0.0


def variant_score(
    left: NormalizedProduct,
    right: NormalizedProduct,
    reasons: list[str],
    contradictions: list[str],
) -> float:
    scores: list[float] = []

    size = size_score(left, right, reasons, contradictions)
    scores.append(size)

    for attr in ("color", "shade", "scent"):
        left_value = getattr(left, attr)
        right_value = getattr(right, attr)
        if left_value and right_value:
            if left_value == right_value:
                reasons.append(f"{attr}_match:{left_value}")
                scores.append(1.0)
            else:
                contradictions.append("clearly_incompatible_variant")
                scores.append(0.0)
        elif left_value or right_value:
            scores.append(0.65)

    if left.pack_count and right.pack_count:
        if left.pack_count == right.pack_count:
            reasons.append(f"pack_match:{left.pack_count}")
            scores.append(1.0)
        else:
            contradictions.append("clearly_incompatible_variant")
            scores.append(0.0)
    elif left.pack_count or right.pack_count:
        reasons.append("pack_missing_one_side")
        scores.append(0.55)

    return sum(scores) / len(scores)


def size_score(
    left: NormalizedProduct,
    right: NormalizedProduct,
    reasons: list[str],
    contradictions: list[str],
) -> float:
    if left.size_value is None and right.size_value is None:
        reasons.append("size_missing_both")
        return 0.55
    if left.size_value is None or right.size_value is None:
        reasons.append("size_missing_one_side")
        return 0.62
    if left.size_ambiguous or right.size_ambiguous:
        reasons.append("size_ambiguous")
        if left.size_unit == right.size_unit and sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit):
            return 0.78
        return 0.55
    if left.size_unit == right.size_unit and sizes_equivalent(left.size_value, left.size_unit, right.size_value, right.size_unit):
        reasons.append(f"size_match:{left.size_value:.0f}{left.size_unit}")
        return 1.0
    contradictions.append("clearly_incompatible_size")
    return 0.0


def category_score(left: NormalizedProduct, right: NormalizedProduct, reasons: list[str]) -> float:
    if not left.category_norm or not right.category_norm:
        return 0.45
    if left.category_norm == right.category_norm:
        reasons.append("category_exact")
        return 1.0
    if left.category_leaf and left.category_leaf == right.category_leaf:
        reasons.append(f"category_leaf:{left.category_leaf}")
        return 0.78
    return string_similarity(left.category_norm, right.category_norm) * 0.8


def specs_description_score(left: NormalizedProduct, right: NormalizedProduct) -> float:
    left_text = " ".join(part for part in [left.specs_text, left.description_norm] if part)
    right_text = " ".join(part for part in [right.specs_text, right.description_norm] if part)
    if not left_text or not right_text:
        return 0.35
    return string_similarity(left_text[:1_500], right_text[:1_500])


def price_score(left: NormalizedProduct, right: NormalizedProduct, reasons: list[str]) -> float:
    if left.price_cents is None or right.price_cents is None:
        return 0.45
    high = max(left.price_cents, right.price_cents)
    low = min(left.price_cents, right.price_cents)
    if high <= 0:
        return 0.0
    ratio = low / high
    if ratio >= 0.85:
        reasons.append("price_close")
        return 1.0
    if ratio >= 0.55:
        return 0.65
    return 0.25
