from __future__ import annotations

from pathlib import Path

import numpy as np

from cartsy_dedupe.embeddings import configured_embedding_dimensions, configured_embedding_model, embedding_provider_name
from cartsy_dedupe.pipeline import (
    DedupePipeline,
    RunMetrics,
    coerce_embedding,
    cosine_similarity,
    embedding_text,
    postgres_retrieval_features,
    should_drop_no_merge_pair,
)
from cartsy_dedupe.utils.pipeline_cache import (
    cache_path_for,
    code_fingerprint,
    embedding_cache_key,
    embedding_text_hash,
    retrieval_layer_cache_key,
    retrieval_rows_to_records,
    write_cache_payload,
    write_embedding_cache,
)
from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.normalize import normalize_row


def test_postgres_retrieval_features_parse_evidence_scores() -> None:
    features = postgres_retrieval_features(
        {
            "exact:ean:123",
            "lexical:fts:0.4500",
            "trigram:title:0.8100",
            "vector:cosine:0.9200",
        }
    )

    assert features["exact"] == 1.0
    assert features["lexical"] > 0.60
    assert features["trigram"] == 0.81
    assert features["vector"] == 0.92


def test_embedding_text_uses_available_product_fields() -> None:
    text = embedding_text(brand="Rhode", title="Peptide Lip Tint", color=None, category="Beauty")

    assert "brand: Rhode" in text
    assert "title: Peptide Lip Tint" in text
    assert "color:" not in text


def test_embedding_provider_configuration_supports_sentence_transformers(monkeypatch) -> None:
    monkeypatch.setenv("CARTSY_EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.delenv("CARTSY_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CARTSY_EMBEDDING_DIMENSIONS", raising=False)

    assert embedding_provider_name() == "sentence-transformers"
    assert configured_embedding_model() == "sentence-transformers/all-MiniLM-L6-v2"
    assert configured_embedding_dimensions() == 384


def test_openai_embedding_dimensions_ignore_stale_dimension_override(monkeypatch) -> None:
    monkeypatch.setenv("CARTSY_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "384")
    monkeypatch.delenv("CARTSY_EMBEDDING_MODEL", raising=False)

    assert configured_embedding_dimensions() == 1536


def test_run_metrics_tracks_openai_usage_and_cost() -> None:
    metrics = RunMetrics()
    metrics.add_usage(
        "text-embedding-3-small",
        {
            "input_tokens": 1_000,
            "total_tokens": 1_000,
            "input_tokens_details": {"cached_tokens": 100},
        },
    )
    report = metrics.as_report(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        input_records=10,
        total_elapsed_seconds=5.0,
    )

    usage = report["openai"]["usage_by_model"]["text-embedding-3-small"]
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 1_000
    assert usage["cached_input_tokens"] == 100
    assert usage["estimated_cost_usd"] > 0
    assert report["timing"]["avg_seconds_per_input_record"] == 0.5


def test_vector_gating_builds_anchor_and_pool_indexes_from_cheap_retrieval(monkeypatch) -> None:
    monkeypatch.setenv("CARTSY_VECTOR_MIN_FTS_RANK", "0.08")
    monkeypatch.setenv("CARTSY_VECTOR_MIN_TRIGRAM_SIMILARITY", "0.60")
    monkeypatch.setenv("CARTSY_VECTOR_INCLUDE_NEIGHBORS", "true")
    pipeline = DedupePipeline()

    profiles = pipeline.build_row_retrieval_profiles(
        exact_rows=[(0, 1, "exact:ean:123")],
        lexical_rows=[(2, 3, "lexical:fts:0.0900"), (4, 5, "lexical:fts:0.0200")],
        trigram_rows=[(2, 6, "trigram:title:0.6100")],
    )

    anchors, pool, stats = pipeline.collect_vector_index_sets(profiles, product_count=8)

    assert anchors == {2, 3, 6}
    assert pool == {2, 3, 6}
    assert stats["vector_anchor_indexes"] == 3
    assert stats["vector_embedding_pool_indexes"] == 3
    assert stats["vector_indexes_skipped_exact"] == 2
    assert stats["vector_indexes_skipped_weak_signal"] == 2
    assert stats["vector_indexes_skipped_no_signal"] == 1


def test_vector_gating_can_expand_pool_with_neighbors(monkeypatch) -> None:
    monkeypatch.setenv("CARTSY_VECTOR_MIN_FTS_RANK", "0.08")
    monkeypatch.setenv("CARTSY_VECTOR_MIN_TRIGRAM_SIMILARITY", "0.60")
    monkeypatch.setenv("CARTSY_VECTOR_INCLUDE_NEIGHBORS", "true")
    pipeline = DedupePipeline()

    profiles = pipeline.build_row_retrieval_profiles(
        exact_rows=[],
        lexical_rows=[(2, 3, "lexical:fts:0.0900"), (3, 4, "lexical:fts:0.0200")],
        trigram_rows=[],
    )

    anchors, pool, stats = pipeline.collect_vector_index_sets(profiles, product_count=5)

    assert anchors == {2, 3}
    assert pool == {2, 3, 4}
    assert stats["vector_indexes_skipped_weak_signal"] == 1


def test_retrieval_layer_cache_reuses_compatible_fts_cache_after_code_key_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CARTSY_STAGE_CACHE_ENABLED", "1")
    pipeline = DedupePipeline()
    layer_params = {"fts_candidates": 25}
    env = {"CARTSY_FTS_CANDIDATES": "25"}
    old_key = retrieval_layer_cache_key(
        normalization_key="norm-key",
        layer="lexical",
        layer_params=layer_params,
        env=env,
        code={"pipeline.py": "old"},
    )
    write_cache_payload(
        cache_path_for("retrieve_candidates_lexical", old_key),
        metadata={
            "stage": "retrieve_candidates:lexical",
            "normalization_key": "norm-key",
            "layer_params": layer_params,
            "env": env,
            "code": {"pipeline.py": "old"},
        },
        payload={"rows": retrieval_rows_to_records([(1, 2, "lexical:fts:0.9000")])},
    )

    def _should_not_fetch(*args, **kwargs):
        raise AssertionError("expected compatible lexical cache to be reused")

    monkeypatch.setattr(pipeline, "fetch_candidate_rows", _should_not_fetch)

    rows = pipeline.load_or_fetch_retrieval_rows(
        object(),
        "lexical",
        sql="select 1",
        params=(25,),
        layer_params=layer_params,
        normalization_key="norm-key",
        retrieval_env=env,
        retrieval_code={"pipeline.py": "new"},
    )

    assert rows == [(1, 2, "lexical:fts:0.9000")]
    assert pipeline.retrieval_layer_cache_status["lexical"]["used"] == 1
    assert pipeline.retrieval_layer_cache_status["lexical"]["mode"] == "compatible"


def test_retrieval_layer_cache_disabled_fetches_and_skips_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CARTSY_STAGE_CACHE_ENABLED", "0")
    pipeline = DedupePipeline()

    monkeypatch.setattr(
        pipeline,
        "fetch_candidate_rows",
        lambda *args, **kwargs: [(1, 2, "trigram:title:0.9000")],
    )

    rows = pipeline.load_or_fetch_retrieval_rows(
        object(),
        "trigram",
        sql="select 1",
        params=(0.55, 25, 500),
        layer_params={"trigram_min_similarity": 0.55, "trigram_candidates": 25, "max_block_size": 500},
        normalization_key="norm-key",
        retrieval_env={"CARTSY_TRIGRAM_CANDIDATES": "25", "CARTSY_TRIGRAM_MIN_SIMILARITY": "0.55"},
        retrieval_code={"pipeline.py": "new"},
    )

    assert rows == [(1, 2, "trigram:title:0.9000")]
    assert pipeline.retrieval_layer_cache_status["trigram"]["enabled"] == 0
    assert not (tmp_path / "cache" / "retrieve_candidates_trigram").exists()


def test_cosine_similarity_handles_dense_semantic_feature() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity(None, [0.0, 1.0]) == 0.0


def test_coerce_embedding_handles_pgvector_string() -> None:
    assert coerce_embedding("[0.1,0.2,0.3]") == [0.1, 0.2, 0.3]
    assert coerce_embedding([0.1, 0.2]) == [0.1, 0.2]


class _FixedModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, rows):
        return [[1.0 - self.score, self.score] for _row in rows]


def _product(**overrides: str):
    row = {
        "id": "1",
        "prod_name": "Cetaphil Locao Hidratante 473ml",
        "brand": "Cetaphil",
        "category": "Beleza>Pele>Hidratantes",
        "description": '["hidratante corporal"]',
        "specs": "{}",
        "img_links": "",
        "url": "",
        "created_at": "",
        "updated_at": "",
        "retailer": "amazon_br",
        "price": "6790",
        "sku": "SKU-1",
        "dimension": "473ml",
    }
    row.update(overrides)
    return normalize_row(row)


def test_score_postgres_pair_uses_logistic_model_and_hard_contradiction() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.95),
        "feature_columns": ["brand_exact", "title_token_set", "semantic_sim", "size_conflict"],
        "threshold": 0.84,
    }
    left = _product(id="1", sku="SHARED", dimension="200ml", prod_name="Cetaphil Locao 200ml")
    right = _product(id="2", sku="SHARED", dimension="473ml", prod_name="Cetaphil Locao 473ml")

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"exact:sku:SHARED"},
        PipelineConfig(merge_threshold=0.84),
        semantic_sim=0.97,
    )

    # CERTAIN_BLOCK (conflicting unambiguous size) bypasses ML entirely.
    assert pair.decision == "no_merge"
    assert pair.score == 0.0
    assert pair.feature_scores["ml_score"] == 0.0
    assert pair.feature_scores["hard_contradiction"] == 1.0


def test_score_postgres_pair_promotes_safe_same_retailer_sku_exact_title() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.0),
        "feature_columns": ["brand_exact", "title_token_set", "semantic_sim", "size_conflict"],
        "threshold": 0.99,
    }
    left = _product(
        id="63956",
        retailer="natura",
        brand="Natura",
        sku="NATBRA-191620",
        prod_name="Base Líquida HD Una 30 ml",
        dimension="30ml",
    )
    right = _product(
        id="64082",
        retailer="natura",
        brand="Natura",
        sku="NATBRA-191620",
        prod_name="Base Líquida HD Una 30 ml",
        dimension="30ml",
    )

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"exact:retailer_sku:natura:natbra 191620"},
        PipelineConfig(merge_threshold=0.99),
        semantic_sim=0.99,
    )

    assert pair.decision == "merge"
    assert pair.score == pair.evidence_score
    assert pair.ml_score == 0.0
    assert pair.decision_threshold == 0.99
    assert 0.95 <= pair.evidence_score < 0.99
    assert pair.decision_reason == "strong_policy:same_retailer_sku_near_exact_title"
    assert "relation:strong_exact_match" in pair.explanation
    assert "strong_policy:same_retailer_sku_near_exact_title" in pair.explanation
    assert "brand_match" in pair.explanation
    assert "title_high" in pair.explanation
    assert "same_retailer_sku" in pair.explanation
    assert "size_match" in pair.explanation
    assert "identifier_match" in pair.explanation
    assert pair.feature_scores["ml_score"] == 0.0


def test_score_postgres_pair_does_not_promote_same_sku_variant_title() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.0),
        "feature_columns": ["brand_exact", "title_token_set", "semantic_sim", "size_conflict"],
        "threshold": 0.99,
    }
    left = _product(
        id="1",
        brand="Commodity",
        sku="PAPER-EXPRESSIVE",
        prod_name="Paper Expressive Eau de Parfum with Sandalwood and Iso E Super",
        dimension="100ml",
    )
    right = _product(
        id="2",
        brand="Commodity",
        sku="PAPER-EXPRESSIVE",
        prod_name="Paper Expressive Eau de Parfum Travel Spray",
        dimension="100ml",
    )

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"exact:retailer_sku:amazon_br:paper expressive"},
        PipelineConfig(merge_threshold=0.99, near_miss_threshold=0.70),
        semantic_sim=0.95,
    )

    assert pair.decision == "no_merge"
    assert pair.score == pair.evidence_score
    assert pair.ml_score == 0.0
    assert 0.0 < pair.evidence_score < 0.70
    assert pair.decision_reason == "below_near_miss_threshold"
    assert "relation:no_match" in pair.explanation
    assert "strong_policy:" not in pair.explanation


def test_score_postgres_pair_blocks_exact_identifier_with_identity_contradiction() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.99),
        "feature_columns": ["brand_exact", "title_token_set", "variant_token_conflict"],
        "threshold": 0.84,
    }
    left = _product(
        id="1",
        brand="M·A·C",
        sku="B08N5WRWNW",
        prod_name="M·A·C Studio Fix Fluid SPF 15 Foundation W4 50ml",
        dimension="50ml",
    )
    right = _product(
        id="2",
        brand="M·A·C",
        retailer="other_shop",
        sku="B08N5WRWNW",
        prod_name="M·A·C Studio Fix Fluid SPF 15 Foundation C7 50ml",
        dimension="50ml",
    )

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"exact:asin:B08N5WRWNW", "lexical:fts:0.9500", "vector:cosine:0.9800"},
        PipelineConfig(merge_threshold=0.84),
        semantic_sim=0.98,
    )

    assert pair.decision == "no_merge"
    assert pair.ml_score == 0.0
    assert pair.decision_reason == "hard_contradiction"
    assert pair.feature_scores["ml_variant_token_conflict"] == 1.0
    assert "variant_token_conflict" in pair.explanation


def test_score_postgres_pair_blocks_exact_identifier_with_one_sided_variant() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.99),
        "feature_columns": ["title_token_set", "variant_token_presence_mismatch"],
        "threshold": 0.84,
    }
    left = _product(
        id="1",
        brand="",
        retailer="mercadolivre",
        sku="MLB2033130474",
        prod_name="Dispositivo Ocular Para Terapia De Luz Vermelha Led 3 Modos",
        dimension="",
    )
    right = _product(
        id="2",
        brand="",
        retailer="mercadolivre",
        sku="MLB2033130474",
        prod_name="Dispositivo Ocular Para Terapia De Luz Vermelha Led 3 Modos Cor Escuro",
        dimension="",
    )

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"exact:retailer_sku:mercadolivre:mlb2033130474", "lexical:fts:0.7143", "trigram:title:1.0000"},
        PipelineConfig(merge_threshold=0.84),
        semantic_sim=0.98,
    )

    assert pair.decision == "no_merge"
    assert pair.decision_reason == "hard_contradiction"
    assert pair.feature_scores["ml_variant_token_presence_mismatch"] == 1.0
    assert "variant_token_presence_mismatch" in pair.explanation


def test_score_postgres_pair_does_not_promote_brand_title_without_exact_sku() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.0),
        "feature_columns": ["brand_exact", "title_token_set", "semantic_sim", "size_conflict"],
        "threshold": 0.99,
    }
    left = _product(
        id="1",
        brand="Natura",
        sku="SKU-1",
        prod_name="Refil Base Líquida HD Una 30 ml NATBRA 173607",
        dimension="30ml",
    )
    right = _product(
        id="2",
        brand="Natura",
        sku="SKU-2",
        prod_name="Refil Base Liquida HD Una 30ml NATBRA 173607",
        dimension="30ml",
    )

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"lexical:fts:0.9800", "trigram:title:0.9800"},
        PipelineConfig(merge_threshold=0.99),
        semantic_sim=0.98,
    )

    assert pair.decision == "no_merge"
    assert pair.score == pair.evidence_score
    assert pair.ml_score == 0.0
    assert 0.0 < pair.evidence_score < 0.70
    assert pair.decision_reason == "below_near_miss_threshold"
    assert "relation:no_match" in pair.explanation
    assert "strong_policy:" not in pair.explanation


def test_score_postgres_pair_blocks_weak_vector_only_ml_merge() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.96),
        "feature_columns": [
            "brand_exact",
            "title_token_set",
            "semantic_sim",
            "size_conflict",
            "retrieval_layer_count",
        ],
        "threshold": 0.84,
    }
    left = _product(
        id="18454",
        brand="genérico",
        sku="B0FGGLCTPS",
        prod_name="Máquina Profissional Dragão Barbeador Acabamento Sem Fio Cabelo Barba Pezinho",
        price="2590",
        dimension="",
    )
    right = _product(
        id="84515",
        brand="Mega",
        retailer="epoca_cosmeticos",
        sku="",
        prod_name="Máquina de Corte Mega Fire I USB-C Bivolt AT61000",
        price="36890",
        dimension="",
    )

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"vector:cosine:0.7806"},
        PipelineConfig(merge_threshold=0.84, evidence_merge_threshold=0.70),
        semantic_sim=0.7806,
    )

    assert pair.decision == "no_merge"
    assert pair.ml_score == 0.96
    assert pair.evidence_score < 0.70
    assert pair.decision_reason == "below_evidence_threshold"
    assert "relation:similar_related_product" in pair.explanation
    assert "evidence_threshold:0.70" in pair.explanation


def test_overconfident_evidence_blocked_pair_is_kept_for_diagnostics() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.96),
        "feature_columns": [
            "brand_exact",
            "title_token_set",
            "semantic_sim",
            "size_conflict",
            "retrieval_layer_count",
        ],
        "threshold": 0.84,
    }
    left = _product(id="1", brand="genérico", sku="B0FGGLCTPS", prod_name="Máquina Profissional Dragão", dimension="")
    right = _product(id="2", brand="Mega", retailer="epoca_cosmeticos", sku="", prod_name="Máquina de Corte Mega", dimension="")

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"vector:cosine:0.7806"},
        PipelineConfig(merge_threshold=0.84, evidence_merge_threshold=0.70),
        semantic_sim=0.7806,
    )

    assert pair.decision == "no_merge"
    assert pair.score < 0.70
    assert should_drop_no_merge_pair(pair, PipelineConfig(near_miss_threshold=0.70)) is False


def test_score_postgres_pair_allows_ml_merge_with_corroborated_evidence() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {
        "model": _FixedModel(0.96),
        "feature_columns": [
            "brand_exact",
            "title_token_set",
            "semantic_sim",
            "size_conflict",
            "retrieval_layer_count",
        ],
        "threshold": 0.84,
    }
    left = _product(id="1", sku="SKU-1")
    right = _product(id="2", retailer="other_shop", sku="SKU-2", price="6990")

    pair = pipeline.score_postgres_pair(
        left,
        right,
        {"lexical:fts:0.7143", "trigram:title:1.0000", "vector:cosine:0.9600"},
        PipelineConfig(merge_threshold=0.84, evidence_merge_threshold=0.70),
        semantic_sim=0.96,
    )

    assert pair.decision == "merge"
    assert pair.ml_score == 0.96
    assert pair.evidence_score >= 0.70
    assert pair.decision_reason == "ml_score_above_threshold"
    assert "relation:candidate_match" in pair.explanation


def test_cli_merge_threshold_is_runtime_floor_for_model_threshold() -> None:
    pipeline = DedupePipeline()
    pipeline.ml_model_bundle = {"threshold": 0.62}

    assert pipeline.ml_threshold(0.84) == 0.84
    assert pipeline.ml_threshold(0.50) == 0.62


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self.conn = conn
        self.rows: list[tuple[str, str, str, str, str, str, str]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        if "SELECT source_id" in sql:
            self.rows = list(self.conn.rows)
            return
        self.conn.executed_sql.append(sql)

    def executemany(self, sql: str, params) -> None:
        self.conn.executed_batches.append((sql, list(params)))

    def fetchall(self) -> list[tuple[str, str, str, str, str, str, str]]:
        return list(self.rows)


class _FakeConn:
    def __init__(self, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
        self.rows = rows
        self.executed_sql: list[str] = []
        self.executed_batches: list[tuple[str, list[tuple[object, str]]]] = []
        self.commit_count = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commit_count += 1


def test_embed_products_reuses_cached_embeddings_without_provider_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CARTSY_EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "3")
    # DedupePipeline loads .env which may disable vector/embeddings globally.
    monkeypatch.setenv("CARTSY_VECTOR_CANDIDATES", "25")
    pipeline = DedupePipeline()
    row = (
        "sku-1",
        "Rhode",
        "Peptide Lip Tint",
        "Beauty",
        "Hydrating tint",
        "color: ribbon",
        "1 pack",
    )
    text = embedding_text(
        brand=row[1],
        title=row[2],
        category=row[3],
        description=row[4],
        specs=row[5],
        dimension=row[6],
    )
    cache_path = (
        Path(tmp_path / "cache")
        / "embeddings"
        / f"{embedding_cache_key(normalization_key='norm-key', embedding_provider=pipeline.embedding_provider, embedding_model=pipeline.embedding_model, embedding_dimensions=pipeline.embedding_dimensions, code=code_fingerprint('utils/pipeline_helpers.py'))}.json"
    )
    write_embedding_cache(
        cache_path,
        entries={
            row[0]: {
                "text_hash": embedding_text_hash(text),
                "embedding": [0.1, 0.2, 0.3],
            }
        },
        metadata={
            "stage": "product_embeddings",
            "normalization_key": "norm-key",
            "embedding_provider": pipeline.embedding_provider,
            "embedding_model": pipeline.embedding_model,
            "embedding_dimensions": pipeline.embedding_dimensions,
            "code": code_fingerprint("utils/pipeline_helpers.py"),
        },
    )

    class _ProviderShouldNotRun:
        def __init__(self, **kwargs) -> None:
            pass

        def embed_texts(self, texts):
            raise AssertionError("expected cached embeddings to skip provider calls")

    monkeypatch.setattr("cartsy_dedupe.pipeline.EmbeddingProvider", _ProviderShouldNotRun)

    conn = _FakeConn([row])
    pipeline.embed_products(conn, normalization_key="norm-key")

    assert pipeline.embedding_cache_hit_count == 1
    assert pipeline.embedding_count == 0
    assert conn.executed_batches == [
        (
            "UPDATE cartsy_products SET embedding = %s WHERE source_id = %s",
            [([0.1, 0.2, 0.3], "sku-1")],
        )
    ]
    assert conn.commit_count == 1


def test_embed_products_reuses_matrix_embedding_cache_without_provider_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CARTSY_EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setenv("CARTSY_VECTOR_CANDIDATES", "25")
    pipeline = DedupePipeline()
    row = (
        "sku-1",
        "Rhode",
        "Peptide Lip Tint",
        "Beauty",
        "Hydrating tint",
        "color: ribbon",
        "1 pack",
    )
    cache_dir = Path(tmp_path / "cache") / "embeddings"
    cache_dir.mkdir(parents=True)
    stem = "embeddings_norm-key_20260430_192710"
    np.save(cache_dir / f"{stem}.npy", np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64))
    (cache_dir / f"{stem}.source_id_to_index.json").write_text('{"sku-1": 0}', encoding="utf-8")

    class _ProviderShouldNotRun:
        def __init__(self, **kwargs) -> None:
            pass

        def embed_texts(self, texts):
            raise AssertionError("expected matrix cached embeddings to skip provider calls")

    monkeypatch.setattr("cartsy_dedupe.pipeline.EmbeddingProvider", _ProviderShouldNotRun)

    conn = _FakeConn([row])
    pipeline.embed_products(conn, normalization_key="norm-key")

    assert pipeline.embedding_cache_hit_count == 1
    assert pipeline.embedding_count == 0
    assert conn.executed_batches == [
        (
            "UPDATE cartsy_products SET embedding = %s WHERE source_id = %s",
            [([0.1, 0.2, 0.3], "sku-1")],
        )
    ]
    assert conn.commit_count == 1


def test_embed_products_skips_cached_embeddings_with_wrong_dimensions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CARTSY_EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setenv("CARTSY_VECTOR_CANDIDATES", "25")
    pipeline = DedupePipeline()
    row = (
        "sku-1",
        "Rhode",
        "Peptide Lip Tint",
        "Beauty",
        "Hydrating tint",
        "color: ribbon",
        "1 pack",
    )
    text = embedding_text(
        brand=row[1],
        title=row[2],
        category=row[3],
        description=row[4],
        specs=row[5],
        dimension=row[6],
    )
    cache_path = (
        Path(tmp_path / "cache")
        / "embeddings"
        / f"{embedding_cache_key(normalization_key='norm-key', embedding_provider=pipeline.embedding_provider, embedding_model=pipeline.embedding_model, embedding_dimensions=pipeline.embedding_dimensions, code=code_fingerprint('utils/pipeline_helpers.py'))}.json"
    )
    write_embedding_cache(
        cache_path,
        entries={
            row[0]: {
                "text_hash": embedding_text_hash(text),
                "embedding": [0.1, 0.2],
            }
        },
        metadata={
            "stage": "product_embeddings",
            "normalization_key": "norm-key",
            "embedding_provider": pipeline.embedding_provider,
            "embedding_model": pipeline.embedding_model,
            "embedding_dimensions": pipeline.embedding_dimensions,
            "code": code_fingerprint("utils/pipeline_helpers.py"),
        },
    )

    class _ProviderReturnsCorrectDimension:
        def __init__(self, **kwargs) -> None:
            pass

        def embed_texts(self, texts):
            class Result:
                embeddings = [[0.4, 0.5, 0.6] for _text in texts]
                usage = None

            return Result()

    monkeypatch.setattr("cartsy_dedupe.pipeline.EmbeddingProvider", _ProviderReturnsCorrectDimension)

    conn = _FakeConn([row])
    pipeline.embed_products(conn, normalization_key="norm-key")

    assert pipeline.embedding_cache_hit_count == 0
    assert pipeline.embedding_count == 1
    assert conn.executed_batches[0] == (
        "UPDATE cartsy_products SET embedding = %s WHERE source_id = %s",
        [([0.4, 0.5, 0.6], "sku-1")],
    )
