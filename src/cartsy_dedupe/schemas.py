from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class NormalizedProduct:
    source_id: str
    retailer: str
    source_sku: str
    url: str
    name_raw: str
    brand_raw: str
    category_raw: str
    description_raw: str
    specs_raw: str
    name_norm: str
    brand_norm: str
    category_norm: str
    category_leaf: str
    description_norm: str
    specs_text: str
    price_cents: int | None
    dimension_raw: str
    size_value: float | None
    size_unit: str | None
    size_ambiguous: bool
    pack_count: int | None
    model_tokens: tuple[str, ...] = field(default_factory=tuple)
    identifiers: dict[str, str] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["model_tokens"] = "|".join(self.model_tokens)
        record["identifiers"] = ";".join(
            f"{key}:{value}" for key, value in sorted(self.identifiers.items())
        )
        record["quality_flags"] = "|".join(self.quality_flags)
        return record


@dataclass(slots=True)
class CandidatePair:
    product_a_id: str
    product_b_id: str
    score: float
    decision: str
    explanation: str
    blocking_keys: tuple[str, ...]
    feature_scores: dict[str, float]
    ml_score: float = 0.0
    evidence_score: float = 0.0
    decision_threshold: float = 0.0
    decision_reason: str = ""

    def to_record(self) -> dict[str, object]:
        return {
            "product_a_id": self.product_a_id,
            "product_b_id": self.product_b_id,
            "score": round(self.score, 4),
            "ml_score": round(self.ml_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "decision_threshold": round(self.decision_threshold, 4),
            "decision_reason": self.decision_reason,
            "decision": self.decision,
            "explanation": self.explanation,
            "blocking_keys": "|".join(self.blocking_keys),
            "feature_scores": ";".join(
                f"{key}:{value:.3f}" for key, value in sorted(self.feature_scores.items())
            ),
        }
