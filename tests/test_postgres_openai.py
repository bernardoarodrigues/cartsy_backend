from __future__ import annotations

from pathlib import Path

from cartsy_dedupe.pipeline import (
    DedupePipeline,
    RunMetrics,
    cosine_similarity,
    embedding_text,
    extracted_attribute_score,
    postgres_retrieval_features,
)
from cartsy_dedupe.utils.pipeline_helpers import ExtractedAttributes
from cartsy_dedupe.utils.pipeline_cache import (
    code_fingerprint,
    embedding_cache_key,
    embedding_text_hash,
    write_embedding_cache,
)


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


def test_extracted_attributes_detect_same_parent_variant_conflict() -> None:
    score, relation, reasons = extracted_attribute_score(
        {"brand": "Rhode", "product_line": "Peptide Lip Treatment", "variant_name": "Salted Caramel"},
        {"brand": "Rhode", "product_line": "Peptide Lip Treatment", "variant_name": "Watermelon Slice"},
    )

    assert score < 1.0
    assert relation == "same_parent_different_variant"
    assert "llm_variant_name_conflict" in reasons


def test_run_metrics_tracks_openai_usage_and_cost() -> None:
    metrics = RunMetrics()
    metrics.add_usage(
        "gpt-5.4-nano",
        {
            "input_tokens": 1_000,
            "output_tokens": 200,
            "total_tokens": 1_200,
            "input_tokens_details": {"cached_tokens": 100},
        },
    )
    report = metrics.as_report(
        embedding_model="text-embedding-3-small",
        extraction_model="gpt-5.4-nano",
        input_records=10,
        total_elapsed_seconds=5.0,
    )

    usage = report["openai"]["usage_by_model"]["gpt-5.4-nano"]
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 1_000
    assert usage["cached_input_tokens"] == 100
    assert usage["output_tokens"] == 200
    assert usage["estimated_cost_usd"] > 0
    assert report["timing"]["avg_seconds_per_input_record"] == 0.5


def test_extracted_attributes_schema_avoids_dynamic_object_fields() -> None:
    schema = ExtractedAttributes.model_json_schema()

    assert "open_attributes" not in schema["properties"]


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


def test_cosine_similarity_handles_dense_semantic_feature() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity(None, [0.0, 1.0]) == 0.0


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


def test_embed_products_reuses_cached_embeddings_without_openai_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
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
        / "all-products"
        / f"{embedding_cache_key(normalization_key='norm-key', embedding_model=pipeline.embedding_model, embedding_dimensions=pipeline.embedding_dimensions, code=code_fingerprint('utils/pipeline_helpers.py'))}.json"
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
            "embedding_model": pipeline.embedding_model,
            "embedding_dimensions": pipeline.embedding_dimensions,
            "code": code_fingerprint("utils/pipeline_helpers.py"),
        },
    )

    class _OpenAIShouldNotRun:
        def __init__(self) -> None:
            self.embeddings = self

        def create(self, **kwargs):
            raise AssertionError("expected cached embeddings to skip OpenAI calls")

    monkeypatch.setattr("cartsy_dedupe.pipeline.OpenAI", _OpenAIShouldNotRun)

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
