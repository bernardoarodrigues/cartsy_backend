from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    merge_threshold: float = 0.84
    ml_model_path: str | None = None
    evidence_merge_threshold: float = 0.78
    near_miss_threshold: float = 0.70
    max_block_size: int | None = 500
    max_candidate_pairs: int | None = 500_000
    near_miss_limit: int = 25_000
    sample_pair_limit: int = 500_000


GLOBAL_IDENTIFIER_KEYS = {"ean", "gtin", "upc"}
MARKETPLACE_IDENTIFIER_KEYS = {"asin"}
GENERIC_BRANDS = {"", "generic", "generico", "genérico"}
