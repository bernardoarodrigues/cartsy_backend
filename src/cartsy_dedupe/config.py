from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    auto_threshold: float = 0.86
    review_threshold: float = 0.70
    max_block_size: int = 1_500
    max_candidate_pairs: int | None = 2_000_000
    review_limit: int = 25_000
    sample_pair_limit: int = 500_000


GLOBAL_IDENTIFIER_KEYS = {"ean", "gtin", "upc"}
MARKETPLACE_IDENTIFIER_KEYS = {"asin"}
GENERIC_BRANDS = {"", "generic", "generico", "genérico"}
