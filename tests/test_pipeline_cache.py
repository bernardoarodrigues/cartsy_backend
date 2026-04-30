from __future__ import annotations

from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct
from cartsy_dedupe.utils.pipeline_cache import (
    CACHE_SCHEMA_VERSION,
    candidate_pairs_from_records,
    candidate_pairs_to_records,
    clustering_cache_key,
    pair_blocks_from_records,
    pair_blocks_to_records,
    product_signature,
    retrieval_cache_key,
    scoring_cache_key,
)


def product(source_id: str) -> NormalizedProduct:
    return NormalizedProduct(
        source_id=source_id,
        retailer="shop",
        source_sku=source_id,
        url=f"https://example.com/{source_id}",
        name_raw="Name",
        brand_raw="Brand",
        category_raw="Category",
        description_raw="Description",
        specs_raw="Specs",
        name_norm="name",
        brand_norm="brand",
        category_norm="category",
        category_leaf="category",
        description_norm="description",
        specs_text="specs",
        price_cents=100,
        dimension_raw="1 pack",
        size_value=1.0,
        size_unit="pack",
        size_ambiguous=False,
        pack_count=1,
    )


def test_pair_blocks_round_trip() -> None:
    pair_blocks = {(1, 2): {"exact:ean:123", "vector:cosine:0.91"}}

    restored = pair_blocks_from_records(pair_blocks_to_records(pair_blocks))

    assert restored == pair_blocks


def test_candidate_pairs_round_trip() -> None:
    candidate_pairs = [
        CandidatePair(
            product_a_id="1",
            product_b_id="2",
            score=0.91,
            decision="merge",
            explanation="ok",
            blocking_keys=("exact:ean:123",),
            feature_scores={"postgres_exact": 1.0},
        )
    ]

    restored = candidate_pairs_from_records(candidate_pairs_to_records(candidate_pairs))

    assert restored == candidate_pairs


def test_product_signature_changes_with_attributes() -> None:
    products = [product("1")]
    signature_before = product_signature(products)
    products[0].extracted_attributes["color"] = "blue"

    assert product_signature(products) != signature_before


def test_stage_cache_keys_change_with_inputs() -> None:
    config = PipelineConfig()
    env = {"OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"}
    code = {"pipeline.py": "abc123"}
    retrieval_key = retrieval_cache_key(
        normalization_key="norm-key",
        config=config,
        env=env,
        code=code,
    )

    assert retrieval_key != retrieval_cache_key(
        normalization_key="other-norm-key",
        config=config,
        env=env,
        code=code,
    )
    assert retrieval_key != retrieval_cache_key(
        normalization_key="norm-key",
        config=PipelineConfig(merge_threshold=0.9),
        env=env,
        code=code,
    )
    assert retrieval_key != retrieval_cache_key(
        normalization_key="norm-key",
        config=config,
        env={"OPENAI_EMBEDDING_MODEL": "text-embedding-3-large"},
        code=code,
    )
    assert retrieval_key != retrieval_cache_key(
        normalization_key="norm-key",
        config=config,
        env=env,
        code={"pipeline.py": "def456"},
    )


def test_downstream_stage_keys_chain_from_parent_stage() -> None:
    config = PipelineConfig()
    scoring_key = scoring_cache_key(retrieval_key="retrieval-a", config=config, code={"scoring.py": "111"})

    assert scoring_key != scoring_cache_key(retrieval_key="retrieval-b", config=config, code={"scoring.py": "111"})
    assert clustering_cache_key(scoring_key=scoring_key, code={"clustering.py": "222"}) != clustering_cache_key(
        scoring_key=scoring_key,
        code={"clustering.py": "333"},
    )
    assert CACHE_SCHEMA_VERSION >= 2
