from __future__ import annotations

from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct
from cartsy_dedupe.utils.pipeline_cache import (
    CACHE_SCHEMA_VERSION,
    candidate_pairs_from_records,
    candidate_pairs_to_records,
    clustering_cache_key,
    embedding_cache_enabled,
    pair_blocks_from_records,
    pair_blocks_to_records,
    product_signature,
    read_cache_payload,
    retrieval_layer_cache_key,
    retrieval_rows_from_records,
    retrieval_rows_to_records,
    retrieval_cache_key,
    scoring_cache_key,
    stage_cache_enabled,
    write_cache_payload,
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


def test_retrieval_rows_round_trip() -> None:
    rows = [(1, 2, "lexical:fts:0.81"), (2, 4, "trigram:title:0.91")]

    restored = retrieval_rows_from_records(retrieval_rows_to_records(rows))

    assert restored == rows


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


def test_product_signature_changes_with_product_content() -> None:
    products = [product("1")]
    signature_before = product_signature(products)
    products[0].quality_flags = ("changed",)

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
    layer_key = retrieval_layer_cache_key(
        normalization_key="norm-key",
        layer="lexical",
        layer_params={"fts_candidates": 25},
        env=env,
        code=code,
    )
    assert layer_key != retrieval_layer_cache_key(
        normalization_key="norm-key",
        layer="trigram",
        layer_params={"fts_candidates": 25},
        env=env,
        code=code,
    )
    assert layer_key != retrieval_layer_cache_key(
        normalization_key="norm-key",
        layer="lexical",
        layer_params={"fts_candidates": 50},
        env=env,
        code=code,
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


def test_stage_cache_toggle_controls_payload_reads_and_writes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "stage.json"
    monkeypatch.setenv("CARTSY_STAGE_CACHE_ENABLED", "false")

    write_cache_payload(path, metadata={"stage": "x"}, payload={"value": 1})

    assert stage_cache_enabled() is False
    assert not path.exists()
    assert read_cache_payload(path) is None

    monkeypatch.setenv("CARTSY_STAGE_CACHE_ENABLED", "true")
    write_cache_payload(path, metadata={"stage": "x"}, payload={"value": 1})

    assert stage_cache_enabled() is True
    assert read_cache_payload(path) == {"value": 1}


def test_embedding_cache_toggle_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("CARTSY_EMBEDDING_CACHE_ENABLED", raising=False)
    assert embedding_cache_enabled() is True

    monkeypatch.setenv("CARTSY_EMBEDDING_CACHE_ENABLED", "0")
    assert embedding_cache_enabled() is False
