"""Postgres + pgvector product deduplication pipeline.

``DedupePipeline`` implements the full retrieval–score–cluster cycle:

1. Exact key blocking (EAN/GTIN/UPC/ASIN/retailer SKU/canonical URL).
2. Lexical full-text search retrieval (``tsvector`` + ``plainto_tsquery``).
3. Trigram similarity blocking (``pg_trgm``).
4. Vector nearest-neighbour retrieval (pgvector HNSW cosine) for products
   with existing lexical or trigram signal.
5. Dense pair embeddings computed for all candidate pairs (semantic_sim feature).
6. Rule-based certainty evaluation (``evaluate_rule``) + logistic-regression
   scoring for uncertain pairs.
7. Union-Find cluster construction with contradiction guard.

``run_pipeline`` orchestrates the pipeline stages, writes run artifacts, and
returns a structured summary report.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from dotenv import load_dotenv
from tqdm import tqdm

from cartsy_dedupe.clustering import build_clusters
from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.embeddings import (
    EmbeddingProvider,
    configured_embedding_dimensions,
    configured_embedding_model,
    embedding_provider_name,
)
from cartsy_dedupe.features import DEFAULT_FEATURE_COLUMNS, build_pair_features, feature_vector, hard_contradiction_features
from cartsy_dedupe.ingest import load_rows
from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.reporting import build_summary_report
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct
from cartsy_dedupe.scoring import MatchCertainty, evaluate_rule
from cartsy_dedupe.storage import prepare_output_dir, write_outputs
from cartsy_dedupe.utils.pipeline_cache import (
    cache_path_for,
    clustering_cache_key,
    code_fingerprint,
    embedding_cache_key,
    embedding_text_hash,
    normalization_cache_dir,
    normalization_cache_key,
    normalize_module_hash,
    product_signature,
    read_embedding_cache,
    read_stage_cache,
    retrieval_layer_cache_key,
    retrieval_rows_from_records,
    retrieval_rows_to_records,
    retrieval_cache_key,
    scoring_cache_key,
    stage_env_fingerprint,
    write_embedding_cache,
    write_stage_cache,
)
from cartsy_dedupe.utils.pipeline_helpers import (
    batched,
    embedding_text,
    exact_keys,
    invert_clusters,
    product_search_text,
)
from cartsy_dedupe.utils.pipeline_metrics import RunMetrics
from cartsy_dedupe.utils.pipeline_sql import (
    evidence_value,
    exact_candidate_sql,
    lexical_candidate_sql,
    postgres_retrieval_features,
    trigram_candidate_sql,
    vector_candidate_sql,
)

try:  # pragma: no cover - import failure is exercised only in incomplete envs.
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:  # pragma: no cover
    psycopg = None
    register_vector = None

try:  # pragma: no cover - import failure is exercised only in incomplete envs.
    from joblib import load as joblib_load
except ImportError:  # pragma: no cover
    joblib_load = None

T = TypeVar("T")
PairBlocks = dict[tuple[int, int], set[str]]
Clusters = dict[str, dict[str, object]]


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        left_float = float(left_value)
        right_float = float(right_value)
        dot += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / ((left_norm**0.5) * (right_norm**0.5))))


def coerce_embedding(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        if not stripped:
            return []
        return [float(part) for part in stripped.split(",") if part.strip()]
    return [float(item) for item in value]


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class RowRetrievalProfile:
    has_exact: bool = False
    fts_hit_count: int = 0
    trigram_hit_count: int = 0
    max_fts_rank: float = 0.0
    max_trigram_similarity: float = 0.0
    neighbor_indexes: set[int] = field(default_factory=set)


class DedupePipeline:
    """Postgres + pgvector implementation of the dedupe architecture."""

    name = "postgres_pgvector"

    def __init__(self, *, dev: bool = False) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        self.database_url = os.getenv("DATABASE_URL", "postgresql://cartsy:cartsy@localhost:5432/cartsy_matcher")
        self.embedding_provider = embedding_provider_name()
        self.embedding_model = configured_embedding_model(self.embedding_provider)
        self.embedding_batch_size = int(os.getenv("CARTSY_EMBEDDING_BATCH_SIZE", "128"))
        self.fts_candidates = int(os.getenv("CARTSY_FTS_CANDIDATES", "25"))
        self.trigram_candidates = int(os.getenv("CARTSY_TRIGRAM_CANDIDATES", "25"))
        self.trigram_min_similarity = float(os.getenv("CARTSY_TRIGRAM_MIN_SIMILARITY", "0.55"))
        self.vector_candidates = int(os.getenv("CARTSY_VECTOR_CANDIDATES", "25"))
        self.vector_require_neighbor = env_flag("CARTSY_VECTOR_REQUIRE_NEIGHBOR", True)
        self.vector_min_fts_rank = float(os.getenv("CARTSY_VECTOR_MIN_FTS_RANK", "0.08"))
        self.vector_min_trigram_similarity = float(os.getenv("CARTSY_VECTOR_MIN_TRIGRAM_SIMILARITY", "0.60"))
        self.vector_include_neighbors = env_flag("CARTSY_VECTOR_INCLUDE_NEIGHBORS", True)
        self.embedding_dimensions = configured_embedding_dimensions(self.embedding_provider, self.embedding_model)
        self.ml_model_bundle: dict[str, Any] | None = None
        self.embedding_count = 0
        self.embedding_cache_hit_count = 0
        self.metrics = RunMetrics()
        self.dev = dev
        self.retrieval_layer_cache_status: dict[str, dict[str, object]] = {}

    def load_ml_model(self, model_path: str | Path | None) -> None:
        if model_path is None:
            raise RuntimeError(
                "The dedupe pipeline now requires a logistic-regression model. "
                "Train one with `cartsy-dedupe train-model ...` and pass `--ml-model`, "
                "or set CARTSY_ML_MODEL_PATH."
            )
        if joblib_load is None:
            raise RuntimeError("Install joblib before loading a logistic-regression model.")
        resolved = Path(model_path)
        if not resolved.is_file():
            raise RuntimeError(f"ML model not found: {resolved}")
        bundle = joblib_load(resolved)
        feature_columns = list(bundle.get("feature_columns") or [])
        missing = [column for column in feature_columns if column not in DEFAULT_FEATURE_COLUMNS]
        if not feature_columns or missing:
            raise RuntimeError(f"ML model has an incompatible feature contract: missing/unknown columns={missing}")
        self.ml_model_bundle = bundle

    def predict_ml_score(self, features: dict[str, float]) -> float:
        if self.ml_model_bundle is None:
            raise RuntimeError(
                "Logistic-regression model is not loaded. Pass --ml-model or set CARTSY_ML_MODEL_PATH."
            )
        columns = list(self.ml_model_bundle.get("feature_columns") or DEFAULT_FEATURE_COLUMNS)
        vector = [feature_vector(features, columns)]
        scaler = self.ml_model_bundle.get("scaler")
        if scaler is not None:
            vector = scaler.transform(vector)
        model = self.ml_model_bundle["model"]
        return float(model.predict_proba(vector)[0][1])

    def ml_threshold(self, fallback: float) -> float:
        if self.ml_model_bundle is None:
            return fallback
        return float(self.ml_model_bundle.get("threshold", fallback))

    def dev_log(self, message: str) -> None:
        if self.dev:
            print(f"[dev] {message}")

    def progress(
        self,
        iterable: Iterable[T],
        *,
        total: int | None = None,
        desc: str,
        unit: str,
        mininterval: float = 0.2,
    ) -> Iterable[T]:
        if not self.dev:
            return iterable
        return tqdm(iterable, total=total, desc=desc, unit=unit, mininterval=mininterval)

    def normalize_rows(self, rows: Iterable[dict[str, str]]) -> list[NormalizedProduct]:
        products: list[NormalizedProduct] = []
        row_count = len(rows) if isinstance(rows, Sequence) else None
        for idx, row in enumerate(
            self.progress(rows, total=row_count, desc="normalize rows", unit="row"),
            start=1,
        ):
            products.append(normalize_row(row))
            if idx % 50_000 == 0:
                print(f"normalized {idx:,} rows")

        self.dev_log("writing normalized rows into Postgres staging tables")
        with self.connect() as conn:
            self.reset_database(conn)
            self.insert_products(conn, products)
            self.insert_exact_keys(conn, products)
        return products

    def generate_candidate_pairs(
        self,
        products: list[NormalizedProduct],
        *,
        config: PipelineConfig,
        normalization_key: str | None = None,
        retrieval_env: dict[str, str | None] | None = None,
        retrieval_code: dict[str, str] | None = None,
    ) -> tuple[PairBlocks, dict[str, int]]:
        self.dev_log("running retrieval stages: exact -> lexical -> trigram -> vector")
        with self.connect() as conn:
            pair_blocks, layer_counts = self.retrieve_candidate_pairs(
                conn,
                config,
                product_count=len(products),
                normalization_key=normalization_key,
                retrieval_env=retrieval_env,
                retrieval_code=retrieval_code,
            )
        stats = {
            "candidate_cap_reached": int(config.max_candidate_pairs is not None and len(pair_blocks) >= config.max_candidate_pairs),
            "exact_pairs": layer_counts.get("exact", 0),
            "lexical_pairs": layer_counts.get("lexical", 0),
            "trigram_pairs": layer_counts.get("trigram", 0),
            "vector_pairs": layer_counts.get("vector", 0),
            "blocking_keys": (
                layer_counts.get("exact", 0)
                + layer_counts.get("lexical", 0)
                + layer_counts.get("trigram", 0)
                + layer_counts.get("vector", 0)
            ),
            "skipped_blocks": 0,
            "oversized_block_rows": 0,
            "embeddings_created": self.embedding_count,
            "cached_embeddings_reused": self.embedding_cache_hit_count,
        }
        stats.update(
            {
                "vector_anchor_indexes": layer_counts.get("vector_anchor_indexes", 0),
                "vector_embedding_pool_indexes": layer_counts.get("vector_embedding_pool_indexes", 0),
                "vector_indexes_skipped_exact": layer_counts.get("vector_indexes_skipped_exact", 0),
                "vector_indexes_skipped_no_signal": layer_counts.get("vector_indexes_skipped_no_signal", 0),
                "vector_indexes_skipped_weak_signal": layer_counts.get("vector_indexes_skipped_weak_signal", 0),
            }
        )
        return pair_blocks, stats

    def score_candidate_pairs(
        self,
        products: list[NormalizedProduct],
        pair_blocks: PairBlocks,
        *,
        config: PipelineConfig,
        normalization_key: str | None = None,
    ) -> tuple[list[CandidatePair], int]:
        candidate_pairs: list[CandidatePair] = []
        semantic_similarities = self.compute_pair_semantic_similarities(
            products,
            pair_blocks,
            normalization_key=normalization_key,
        )
        pair_items = pair_blocks.items()
        for pair_number, ((left_index, right_index), block_keys) in enumerate(
            self.progress(pair_items, total=len(pair_blocks), desc="score pairs", unit="pair"),
            start=1,
        ):
            left = products[left_index]
            right = products[right_index]
            pair = self.score_postgres_pair(
                left,
                right,
                block_keys,
                config,
                semantic_sim=semantic_similarities.get((left_index, right_index), 0.0),
            )
            if pair.decision == "no_merge" and pair.score < config.near_miss_threshold:
                continue
            candidate_pairs.append(pair)
            if pair_number % 100_000 == 0:
                print(f"scored {pair_number:,} candidate pairs; kept {len(candidate_pairs):,}")
        return candidate_pairs, len(pair_blocks)

    def compute_pair_semantic_similarities(
        self,
        products: list[NormalizedProduct],
        pair_blocks: PairBlocks,
        *,
        normalization_key: str | None = None,
    ) -> dict[tuple[int, int], float]:
        if not pair_blocks:
            return {}
        pair_product_indexes = {index for pair in pair_blocks for index in pair}
        self.dev_log(f"embedding {len(pair_product_indexes):,} candidate-pair products for dense semantic features")
        with self.connect() as conn:
            self.embed_products(
                conn,
                only_indexes=pair_product_indexes,
                normalization_key=normalization_key,
                force=True,
            )
            embeddings = self.fetch_embeddings_by_index(conn, pair_product_indexes)
        missing = len(pair_product_indexes - set(embeddings))
        if missing:
            self.dev_log(f"semantic features missing embeddings for {missing:,} candidate-pair products")
        similarities: dict[tuple[int, int], float] = {}
        for left_index, right_index in pair_blocks:
            left_embedding = embeddings.get(left_index)
            right_embedding = embeddings.get(right_index)
            similarities[(left_index, right_index)] = cosine_similarity(left_embedding, right_embedding)
        return similarities

    def fetch_embeddings_by_index(self, conn, indexes: set[int]) -> dict[int, list[float]]:
        if not indexes:
            return {}
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_index, embedding
                FROM cartsy_products
                WHERE source_index = ANY(%s)
                  AND embedding IS NOT NULL
                """,
                (sorted(indexes),),
            )
            rows = cur.fetchall()
        return {int(index): coerce_embedding(embedding) for index, embedding in rows}

    def build_clusters(
        self,
        products: list[NormalizedProduct],
        candidate_pairs: list[CandidatePair],
        id_to_index: dict[str, int],
    ) -> tuple[Clusters, dict[str, int]]:
        return build_clusters(products, candidate_pairs, id_to_index)

    def build_summary_report(
        self,
        *,
        products: list[NormalizedProduct],
        candidate_pairs: list[CandidatePair],
        clusters: Clusters,
        blocking_stats: dict[str, int],
        cluster_stats: dict[str, int],
        scored_candidate_pairs: int,
        elapsed_seconds: float,
    ) -> dict[str, object]:
        report = build_summary_report(
            products=products,
            candidate_pairs=candidate_pairs,
            clusters=clusters,
            blocking_stats=blocking_stats,
            cluster_stats=cluster_stats,
            scored_candidate_pairs=scored_candidate_pairs,
            elapsed_seconds=elapsed_seconds,
        )
        report["pipeline"] = self.name
        report["architecture_notes"] = {
            "database": "Postgres with pg_trgm, full-text search, and pgvector",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "cascade": "exact keys -> full-text retrieval -> trigram retrieval -> vector retrieval -> dense pair embeddings -> logistic regression",
        }
        return report

    def connect(self):
        if psycopg is None:
            raise RuntimeError("Install psycopg[binary] and pgvector before running the Postgres pipeline.")
        try:
            conn = psycopg.connect(self.database_url)
        except Exception as exc:  # pragma: no cover - depends on local service state.
            raise RuntimeError(
                "Could not connect to Postgres. Start it with `docker compose up -d postgres` "
                "or set DATABASE_URL to a reachable pgvector database."
            ) from exc
        return conn

    def reset_database(self, conn) -> None:
        if register_vector is None:
            raise RuntimeError("Install pgvector before running the Postgres pipeline.")
        vector_type = f"vector({self.embedding_dimensions})"
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        conn.commit()
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS cartsy_exact_keys")
            cur.execute("DROP TABLE IF EXISTS cartsy_products")
            cur.execute(
                f"""
                CREATE TABLE cartsy_products (
                    source_id TEXT PRIMARY KEY,
                    source_index INTEGER NOT NULL UNIQUE,
                    retailer TEXT NOT NULL,
                    source_sku TEXT NOT NULL,
                    url TEXT NOT NULL,
                    name_raw TEXT NOT NULL,
                    brand_raw TEXT NOT NULL,
                    category_raw TEXT NOT NULL,
                    description_raw TEXT NOT NULL,
                    specs_raw TEXT NOT NULL,
                    name_norm TEXT NOT NULL,
                    brand_norm TEXT NOT NULL,
                    category_norm TEXT NOT NULL,
                    category_leaf TEXT NOT NULL,
                    description_norm TEXT NOT NULL,
                    specs_text TEXT NOT NULL,
                    price_cents INTEGER,
                    dimension_raw TEXT NOT NULL,
                    size_value DOUBLE PRECISION,
                    size_unit TEXT,
                    size_ambiguous BOOLEAN NOT NULL,
                    pack_count INTEGER,
                    model_tokens TEXT[] NOT NULL,
                    identifiers JSONB NOT NULL,
                    quality_flags TEXT[] NOT NULL,
                    search_text TEXT NOT NULL,
                    search_vector TSVECTOR,
                    embedding {vector_type}
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE cartsy_exact_keys (
                    product_index INTEGER NOT NULL REFERENCES cartsy_products(source_index) ON DELETE CASCADE,
                    key_type TEXT NOT NULL,
                    key_value TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX idx_cartsy_products_brand ON cartsy_products (brand_norm)")
            cur.execute("CREATE INDEX idx_cartsy_products_search ON cartsy_products USING GIN (search_vector)")
            cur.execute("CREATE INDEX idx_cartsy_products_title_trgm ON cartsy_products USING GIN (name_norm gin_trgm_ops)")
            cur.execute("CREATE INDEX idx_cartsy_products_title_trgm_gist ON cartsy_products USING GiST (name_norm gist_trgm_ops)")
            cur.execute("CREATE INDEX idx_cartsy_exact_keys ON cartsy_exact_keys (key_type, key_value)")
        conn.commit()

    def insert_products(self, conn, products: list[NormalizedProduct]) -> None:
        rows = []
        for index, product in enumerate(products):
            record = asdict(product)
            search_text = product_search_text(product)
            rows.append(
                (
                    product.source_id,
                    index,
                    product.retailer,
                    product.source_sku,
                    product.url,
                    product.name_raw,
                    product.brand_raw,
                    product.category_raw,
                    product.description_raw,
                    product.specs_raw,
                    product.name_norm,
                    product.brand_norm,
                    product.category_norm,
                    product.category_leaf,
                    product.description_norm,
                    product.specs_text,
                    product.price_cents,
                    product.dimension_raw,
                    product.size_value,
                    product.size_unit,
                    product.size_ambiguous,
                    product.pack_count,
                    list(product.model_tokens),
                    json.dumps(record["identifiers"], ensure_ascii=False),
                    list(product.quality_flags),
                    search_text,
                )
            )
        with conn.cursor() as cur:
            with cur.copy(
                """
                COPY cartsy_products (
                    source_id, source_index, retailer, source_sku, url, name_raw, brand_raw,
                    category_raw, description_raw, specs_raw, name_norm, brand_norm,
                    category_norm, category_leaf, description_norm, specs_text, price_cents,
                    dimension_raw, size_value, size_unit, size_ambiguous, pack_count,
                    model_tokens, identifiers, quality_flags, search_text
                ) FROM STDIN
                """
            ) as copy:
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                """
                UPDATE cartsy_products
                SET search_vector =
                    setweight(to_tsvector('simple', coalesce(brand_norm, '')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(name_norm, '')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(category_norm, '')), 'B') ||
                    setweight(to_tsvector('simple', coalesce(specs_text, '')), 'C') ||
                    setweight(to_tsvector('simple', coalesce(description_norm, '')), 'C')
                """
            )
        conn.commit()

    def insert_exact_keys(self, conn, products: list[NormalizedProduct]) -> None:
        rows: list[tuple[int, str, str]] = []
        for index, product in enumerate(products):
            for key, value in exact_keys(product).items():
                rows.append((index, key, value))
        with conn.cursor() as cur:
            with cur.copy("COPY cartsy_exact_keys (product_index, key_type, key_value) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
        conn.commit()

    def embed_products(
        self,
        conn,
        *,
        only_indexes: set[int] | None = None,
        exclude_indexes: set[int] | None = None,
        normalization_key: str | None = None,
        force: bool = False,
    ) -> None:
        if self.vector_candidates <= 0 and not force:
            self.dev_log("skipping embedding generation because CARTSY_VECTOR_CANDIDATES <= 0")
            return
        embedder = EmbeddingProvider(provider=self.embedding_provider, model=self.embedding_model)
        only_indexes = only_indexes or set()
        exclude_indexes = exclude_indexes or set()
        if only_indexes:
            allowed_indexes = sorted(only_indexes - exclude_indexes)
            if not allowed_indexes:
                return
        with conn.cursor() as cur:
            if only_indexes:
                cur.execute(
                    """
                    SELECT source_id, brand_raw, name_raw, category_raw, description_raw, specs_raw,
                           dimension_raw
                    FROM cartsy_products
                    WHERE embedding IS NULL
                      AND source_index = ANY(%s)
                    ORDER BY source_index
                    """,
                    (allowed_indexes,),
                )
            elif exclude_indexes:
                cur.execute(
                    """
                    SELECT source_id, brand_raw, name_raw, category_raw, description_raw, specs_raw,
                           dimension_raw
                    FROM cartsy_products
                    WHERE embedding IS NULL
                      AND NOT (source_index = ANY(%s))
                    ORDER BY source_index
                    """,
                    (sorted(exclude_indexes),),
                )
            else:
                cur.execute(
                    """
                    SELECT source_id, brand_raw, name_raw, category_raw, description_raw, specs_raw,
                           dimension_raw
                    FROM cartsy_products
                    WHERE embedding IS NULL
                    ORDER BY source_index
                    """
                )
            rows = cur.fetchall()
        if not rows:
            return

        embedding_cache_path: Path | None = None
        embedding_cache_entries: dict[str, dict[str, Any]] = {}
        embedding_cache_metadata: dict[str, Any] | None = None
        if normalization_key:
            embedding_code = code_fingerprint("utils/pipeline_helpers.py")
            embedding_cache_id = embedding_cache_key(
                normalization_key=normalization_key,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                embedding_dimensions=self.embedding_dimensions,
                code=embedding_code,
            )
            embedding_cache_path = cache_path_for("embeddings", embedding_cache_id)
            embedding_cache_entries = read_embedding_cache(embedding_cache_path) or {}
            embedding_cache_metadata = {
                "stage": "product_embeddings",
                "normalization_key": normalization_key,
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
                "code": embedding_code,
            }

        rows_to_embed: list[tuple[str, str, str, str, str, str, str]] = []
        cached_updates: list[tuple[list[float], str]] = []
        for row in rows:
            text = embedding_text(
                brand=row[1],
                title=row[2],
                category=row[3],
                description=row[4],
                specs=row[5],
                dimension=row[6],
            )
            cached_entry = embedding_cache_entries.get(row[0])
            if cached_entry and cached_entry.get("text_hash") == embedding_text_hash(text):
                cached_updates.append((list(cached_entry["embedding"]), row[0]))
                continue
            rows_to_embed.append(row)
        if cached_updates:
            with conn.cursor() as cur:
                cur.executemany("UPDATE cartsy_products SET embedding = %s WHERE source_id = %s", cached_updates)
            self.embedding_cache_hit_count += len(cached_updates)
            self.dev_log(f"reused {len(cached_updates):,} cached embeddings")

        batches = list(batched(rows_to_embed, self.embedding_batch_size))
        self.dev_log(f"creating embeddings in {len(batches):,} batches")
        for batch in self.progress(
            batches,
            total=len(batches),
            desc="embed batches",
            unit="batch",
        ):
            texts = [
                embedding_text(
                    brand=row[1],
                    title=row[2],
                    category=row[3],
                    description=row[4],
                    specs=row[5],
                    dimension=row[6],
                )
                for row in batch
            ]
            result = embedder.embed_texts(texts)
            if result.usage is not None:
                self.metrics.add_usage(self.embedding_model, result.usage)
            updates = [(embedding, row[0]) for embedding, row in zip(result.embeddings, batch, strict=True)]
            with conn.cursor() as cur:
                cur.executemany("UPDATE cartsy_products SET embedding = %s WHERE source_id = %s", updates)
            self.embedding_count += len(updates)
            print(f"embedded {self.embedding_count:,} products")
            if embedding_cache_path is not None and embedding_cache_metadata is not None:
                for embedding, row, text in zip(result.embeddings, batch, texts, strict=True):
                    embedding_cache_entries[row[0]] = {
                        "text_hash": embedding_text_hash(text),
                        "embedding": embedding,
                    }
                write_embedding_cache(
                    embedding_cache_path,
                    entries=embedding_cache_entries,
                    metadata=embedding_cache_metadata,
                )
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cartsy_products_embedding
                ON cartsy_products USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL
                """
            )
        conn.commit()

    def retrieve_candidate_pairs(
        self,
        conn,
        config: PipelineConfig,
        *,
        product_count: int,
        normalization_key: str | None = None,
        retrieval_env: dict[str, str | None] | None = None,
        retrieval_code: dict[str, str] | None = None,
    ) -> tuple[PairBlocks, Counter[str]]:
        pairs: PairBlocks = defaultdict(set)
        counts: Counter[str] = Counter()
        self.dev_log("retrieval stage: exact keys")
        exact_rows = self.load_or_fetch_retrieval_rows(
            conn,
            layer="exact",
            sql=exact_candidate_sql(),
            params=(),
            layer_params={},
            normalization_key=normalization_key,
            retrieval_env=retrieval_env,
            retrieval_code=retrieval_code,
        )
        self.merge_candidate_rows(pairs, counts, "exact", exact_rows, config.max_candidate_pairs)
        exact_resolved_indexes = {
            index
            for pair, evidence in pairs.items()
            if any(key.startswith("exact:") for key in evidence)
            for index in pair
        }
        self.dev_log("retrieval stage: lexical FTS")
        lexical_rows = self.load_or_fetch_retrieval_rows(
            conn,
            "lexical",
            sql=lexical_candidate_sql(),
            params=(self.fts_candidates,),
            layer_params={"fts_candidates": self.fts_candidates},
            normalization_key=normalization_key,
            retrieval_env=retrieval_env,
            retrieval_code=retrieval_code,
        )
        self.merge_candidate_rows(pairs, counts, "lexical", lexical_rows, config.max_candidate_pairs)
        self.dev_log("retrieval stage: trigram")
        trigram_rows = self.load_or_fetch_retrieval_rows(
            conn,
            "trigram",
            sql=trigram_candidate_sql(),
            params=(self.trigram_min_similarity, self.trigram_candidates, config.max_block_size),
            layer_params={
                "trigram_min_similarity": self.trigram_min_similarity,
                "trigram_candidates": self.trigram_candidates,
                "max_block_size": config.max_block_size,
            },
            normalization_key=normalization_key,
            retrieval_env=retrieval_env,
            retrieval_code=retrieval_code,
        )
        self.merge_candidate_rows(pairs, counts, "trigram", trigram_rows, config.max_candidate_pairs)
        anchor_indexes: set[int] = set()
        embedding_pool_indexes: set[int] = set()
        vector_gating_stats: Counter[str] = Counter()
        if self.vector_candidates > 0:
            profiles = self.build_row_retrieval_profiles(exact_rows, lexical_rows, trigram_rows)
            anchor_indexes, embedding_pool_indexes, vector_gating_stats = self.collect_vector_index_sets(
                profiles,
                product_count=product_count,
            )
        cap_reached = config.max_candidate_pairs is not None and len(pairs) >= config.max_candidate_pairs
        if self.vector_candidates > 0 and anchor_indexes and not cap_reached:
            self.dev_log("retrieval stage: vector embeddings")
            self.dev_log(
                "vector gating kept "
                f"{len(anchor_indexes):,} anchor rows and {len(embedding_pool_indexes):,} embedding-pool rows"
            )
            self.embed_products(
                conn,
                only_indexes=embedding_pool_indexes,
                exclude_indexes=exact_resolved_indexes,
                normalization_key=normalization_key,
            )
            vector_rows = self.load_or_fetch_retrieval_rows(
                conn,
                "vector",
                sql=vector_candidate_sql(),
                params=(sorted(embedding_pool_indexes), self.vector_candidates, sorted(anchor_indexes)),
                layer_params={
                    "vector_candidates": self.vector_candidates,
                    "excluded_exact_index_count": len(exact_resolved_indexes),
                    "vector_anchor_indexes": len(anchor_indexes),
                    "vector_embedding_pool_indexes": len(embedding_pool_indexes),
                },
                normalization_key=normalization_key,
                retrieval_env=retrieval_env,
                retrieval_code=retrieval_code,
            )
            self.merge_candidate_rows(pairs, counts, "vector", vector_rows, config.max_candidate_pairs)
        counts.update(vector_gating_stats)
        return pairs, counts

    def build_row_retrieval_profiles(
        self,
        exact_rows: list[tuple[int, int, str]],
        lexical_rows: list[tuple[int, int, str]],
        trigram_rows: list[tuple[int, int, str]],
    ) -> dict[int, RowRetrievalProfile]:
        profiles: dict[int, RowRetrievalProfile] = defaultdict(RowRetrievalProfile)
        for left, right, _evidence in exact_rows:
            profiles[left].has_exact = True
            profiles[right].has_exact = True
        for left, right, evidence in lexical_rows:
            rank = evidence_value(evidence, default=0.0)
            left_profile = profiles[left]
            right_profile = profiles[right]
            left_profile.fts_hit_count += 1
            right_profile.fts_hit_count += 1
            left_profile.max_fts_rank = max(left_profile.max_fts_rank, rank)
            right_profile.max_fts_rank = max(right_profile.max_fts_rank, rank)
            left_profile.neighbor_indexes.add(right)
            right_profile.neighbor_indexes.add(left)
        for left, right, evidence in trigram_rows:
            similarity = evidence_value(evidence, default=0.0)
            left_profile = profiles[left]
            right_profile = profiles[right]
            left_profile.trigram_hit_count += 1
            right_profile.trigram_hit_count += 1
            left_profile.max_trigram_similarity = max(left_profile.max_trigram_similarity, similarity)
            right_profile.max_trigram_similarity = max(right_profile.max_trigram_similarity, similarity)
            left_profile.neighbor_indexes.add(right)
            right_profile.neighbor_indexes.add(left)
        return dict(profiles)

    def collect_vector_index_sets(
        self,
        profiles: dict[int, RowRetrievalProfile],
        *,
        product_count: int,
    ) -> tuple[set[int], set[int], Counter[str]]:
        anchor_indexes: set[int] = set()
        embedding_pool_indexes: set[int] = set()
        stats: Counter[str] = Counter()
        for index in range(product_count):
            profile = profiles.get(index)
            if profile is None:
                stats["vector_indexes_skipped_no_signal"] += 1
                continue
            if profile.has_exact:
                stats["vector_indexes_skipped_exact"] += 1
                continue
            if self.vector_require_neighbor and not profile.neighbor_indexes:
                stats["vector_indexes_skipped_no_signal"] += 1
                continue
            strong_fts = profile.max_fts_rank >= self.vector_min_fts_rank
            strong_trigram = profile.max_trigram_similarity >= self.vector_min_trigram_similarity
            multi_signal = profile.fts_hit_count > 0 and profile.trigram_hit_count > 0
            if not (strong_fts or strong_trigram or multi_signal):
                stats["vector_indexes_skipped_weak_signal"] += 1
                continue
            anchor_indexes.add(index)
            embedding_pool_indexes.add(index)
            if self.vector_include_neighbors:
                for neighbor in profile.neighbor_indexes:
                    neighbor_profile = profiles.get(neighbor)
                    if neighbor_profile is None or not neighbor_profile.has_exact:
                        embedding_pool_indexes.add(neighbor)
        stats["vector_anchor_indexes"] = len(anchor_indexes)
        stats["vector_embedding_pool_indexes"] = len(embedding_pool_indexes)
        return anchor_indexes, embedding_pool_indexes, stats

    def load_or_fetch_retrieval_rows(
        self,
        conn,
        layer: str,
        *,
        sql: str,
        params: tuple[object, ...],
        layer_params: dict[str, object],
        normalization_key: str | None,
        retrieval_env: dict[str, str | None] | None,
        retrieval_code: dict[str, str] | None,
    ) -> list[tuple[int, int, str]]:
        layer_env = self.layer_cache_env(layer, retrieval_env) if retrieval_env is not None else None
        if normalization_key and layer_env is not None and retrieval_code is not None:
            layer_key = retrieval_layer_cache_key(
                normalization_key=normalization_key,
                layer=layer,
                layer_params=layer_params,
                env=layer_env,
                code=retrieval_code,
            )
            layer_path = cache_path_for(f"retrieve_candidates_{layer}", layer_key)
            self.retrieval_layer_cache_status[layer] = {
                "used": 0,
                "path": str(layer_path),
                "key": layer_key,
            }
            cached = read_stage_cache(layer_path)
            if cached is not None:
                self.retrieval_layer_cache_status[layer]["used"] = 1
                return retrieval_rows_from_records(cached["payload"].get("rows") or [])
            fallback_path, fallback_blob = self.find_compatible_retrieval_layer_cache(
                layer=layer,
                layer_path=layer_path,
                normalization_key=normalization_key,
                layer_params=layer_params,
            )
            if fallback_blob is not None:
                self.retrieval_layer_cache_status[layer]["used"] = 1
                self.retrieval_layer_cache_status[layer]["mode"] = "lenient"
                self.retrieval_layer_cache_status[layer]["path"] = str(fallback_path)
                return retrieval_rows_from_records(fallback_blob["payload"].get("rows") or [])

        rows = self.fetch_candidate_rows(conn, layer, sql, params)
        if normalization_key and layer_env is not None and retrieval_code is not None:
            write_stage_cache(
                layer_path,
                metadata={
                    "stage": f"retrieve_candidates:{layer}",
                    "normalization_key": normalization_key,
                    "layer_params": layer_params,
                    "env": layer_env,
                    "code": retrieval_code,
                },
                payload={"rows": retrieval_rows_to_records(rows)},
            )
        return rows

    @staticmethod
    def layer_cache_env(layer: str, retrieval_env: dict[str, str | None]) -> dict[str, str | None]:
        env_keys_by_layer = {
            "exact": tuple(),
            "lexical": ("CARTSY_FTS_CANDIDATES",),
            "trigram": ("CARTSY_TRIGRAM_CANDIDATES", "CARTSY_TRIGRAM_MIN_SIMILARITY"),
            "vector": (
                "OPENAI_EMBEDDING_MODEL",
                "CARTSY_EMBEDDING_DIMENSIONS",
                "CARTSY_EMBEDDING_BATCH_SIZE",
                "CARTSY_VECTOR_CANDIDATES",
                "CARTSY_VECTOR_REQUIRE_NEIGHBOR",
                "CARTSY_VECTOR_MIN_FTS_RANK",
                "CARTSY_VECTOR_MIN_TRIGRAM_SIMILARITY",
                "CARTSY_VECTOR_INCLUDE_NEIGHBORS",
            ),
        }
        keys = env_keys_by_layer.get(layer)
        if keys is None:
            return dict(retrieval_env)
        return {key: retrieval_env.get(key) for key in keys}

    def find_compatible_retrieval_layer_cache(
        self,
        *,
        layer: str,
        layer_path: Path,
        normalization_key: str,
        layer_params: dict[str, object],
    ) -> tuple[Path | None, dict[str, Any] | None]:
        stage_name = f"retrieve_candidates:{layer}"
        for candidate_path in sorted(layer_path.parent.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            if candidate_path == layer_path:
                continue
            candidate_blob = read_stage_cache(candidate_path)
            if candidate_blob is None:
                continue
            metadata = candidate_blob.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if not self.cache_metadata_matches_layer(
                metadata=metadata,
                stage_name=stage_name,
                normalization_key=normalization_key,
                layer_params=layer_params,
            ):
                continue
            return candidate_path, candidate_blob
        return None, None

    @staticmethod
    def cache_metadata_matches_layer(
        *,
        metadata: dict[str, Any],
        stage_name: str,
        normalization_key: str,
        layer_params: dict[str, object],
    ) -> bool:
        return (
            metadata.get("stage") == stage_name
            and metadata.get("normalization_key") == normalization_key
            and metadata.get("layer_params") == layer_params
        )

    def fetch_candidate_rows(
        self,
        conn,
        layer: str,
        sql: str,
        params: tuple[object, ...],
    ) -> list[tuple[int, int, str]]:
        show_progress = layer in {"lexical", "trigram", "vector"}
        cursor_name = f"cartsy_{layer}_{int(perf_counter() * 1_000_000)}"
        collected: list[tuple[int, int, str]] = []
        with conn.cursor(name=cursor_name) as cur:
            cur.execute(sql, params)
            bar = tqdm(desc=f"{layer} retrieval rows", unit="row", mininterval=0.2) if show_progress else None
            try:
                while True:
                    rows = cur.fetchmany(2_000)
                    if not rows:
                        break
                    if bar is not None:
                        bar.update(len(rows))
                    for left, right, evidence in rows:
                        if left == right:
                            continue
                        if left > right:
                            left, right = right, left
                        collected.append((left, right, str(evidence)))
            finally:
                if bar is not None:
                    bar.close()
        return collected

    def merge_candidate_rows(
        self,
        pairs: PairBlocks,
        counts: Counter[str],
        layer: str,
        rows: list[tuple[int, int, str]],
        max_candidate_pairs: int | None,
    ) -> None:
        if max_candidate_pairs is not None and len(pairs) >= max_candidate_pairs:
            return
        for left, right, evidence in rows:
            pairs[(left, right)].add(evidence)
            counts[layer] += 1
            if max_candidate_pairs is not None and len(pairs) >= max_candidate_pairs:
                return

    def score_postgres_pair(
        self,
        left: NormalizedProduct,
        right: NormalizedProduct,
        block_keys: set[str],
        config: PipelineConfig,
        *,
        semantic_sim: float = 0.0,
    ) -> CandidatePair:
        rule_decision = evaluate_rule(left, right)
        retrieval = postgres_retrieval_features(block_keys)
        pair_features = build_pair_features(
            left, right, block_keys,
            semantic_sim=semantic_sim,
            rule_decision=rule_decision,
        )
        threshold = self.ml_threshold(config.merge_threshold)

        if rule_decision.certainty == MatchCertainty.CERTAIN_MATCH:
            ml_score, score, decision = 1.0, 1.0, "merge"
            relation = "certain_match"
            hard_contradiction_val = 0.0
        elif rule_decision.certainty == MatchCertainty.CERTAIN_BLOCK:
            ml_score, score, decision = 0.0, 0.0, "no_merge"
            relation = "certain_block"
            hard_contradiction_val = 1.0
        else:
            ml_score = self.predict_ml_score(pair_features)
            hard_contradiction = hard_contradiction_features(pair_features)
            hard_contradiction_val = float(hard_contradiction)
            score = ml_score
            relation = "exact_match"
            if hard_contradiction:
                relation = "same_parent_different_variant"
                score = min(ml_score, threshold - 0.01)
            elif ml_score < threshold and ml_score >= config.near_miss_threshold:
                relation = "similar_related_product"
            elif ml_score < config.near_miss_threshold:
                relation = "no_match"
            score = max(0.0, min(1.0, score))
            decision = "merge" if not hard_contradiction and ml_score >= threshold else "no_merge"

        explanations = [
            f"relation:{relation}",
            f"rule:{rule_decision.certainty.value}",
            f"rule_reason:{rule_decision.reason}",
            f"ml_score:{ml_score:.2f}",
            f"ml_threshold:{threshold:.2f}",
            f"exact:{retrieval['exact']:.2f}",
            f"fts:{retrieval['lexical']:.2f}",
            f"trigram:{retrieval['trigram']:.2f}",
            f"vector:{retrieval['vector']:.2f}",
            f"semantic:{semantic_sim:.2f}",
        ]
        feature_scores = {
            **rule_decision.feature_scores,
            "postgres_exact": retrieval["exact"],
            "postgres_fts": retrieval["lexical"],
            "postgres_trigram": retrieval["trigram"],
            "postgres_vector": retrieval["vector"],
            "semantic_sim": semantic_sim,
            "ml_score": ml_score,
            "ml_threshold": threshold,
            "hard_contradiction": hard_contradiction_val,
        }
        feature_scores.update({f"ml_{key}": value for key, value in pair_features.items()})
        return CandidatePair(
            product_a_id=left.source_id,
            product_b_id=right.source_id,
            score=score,
            decision=decision,
            explanation="; ".join(explanations[:18]),
            blocking_keys=tuple(sorted(block_keys)),
            feature_scores=feature_scores,
        )


def run_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    limit: int | None = None,
    dev: bool = False,
) -> dict[str, object]:
    run_started_at = datetime.now(timezone.utc)
    started = perf_counter()
    resolved_input_path = Path(input_path).resolve()
    output_path = prepare_output_dir(output_dir)
    dedupe_pipeline = DedupePipeline(dev=dev)
    dedupe_pipeline.load_ml_model(config.ml_model_path)
    stage_timeline: list[dict[str, object]] = []

    def run_stage(name: str, action, *, items: int = 0):
        print(f"starting stage: {name}")
        stage_started = perf_counter()
        stage_started_at = datetime.now(timezone.utc)
        try:
            with dedupe_pipeline.metrics.stage(name, items=items):
                return action()
        finally:
            stage_ended_at = datetime.now(timezone.utc)
            stage_elapsed = perf_counter() - stage_started
            stage_timeline.append(
                {
                    "name": name,
                    "started_at_utc": stage_started_at.isoformat(),
                    "ended_at_utc": stage_ended_at.isoformat(),
                    "elapsed_seconds": round(stage_elapsed, 3),
                }
            )
            print(f"finished stage: {name} ({stage_elapsed:.2f}s)")

    normalize_hash = normalize_module_hash()
    cache_key = normalization_cache_key(input_path=resolved_input_path, limit=limit, normalize_hash=normalize_hash)
    cache_path = normalization_cache_dir() / f"{cache_key}.json"
    stage_cache_status: dict[str, dict[str, object]] = {
        "normalize_and_load_postgres": {"used": 0, "path": str(cache_path), "key": cache_key},
    }

    print(f"loading {input_path}")
    dedupe_pipeline.dev_log("stage start: load_rows")

    def load_rows_action():
        return load_rows(input_path, limit=limit)

    rows = run_stage("load_rows", load_rows_action)
    print(f"loaded {len(rows):,} rows")

    print("normalizing and loading Postgres")
    dedupe_pipeline.dev_log("stage start: normalize_and_load_postgres")

    def normalize_and_load_action():
        return dedupe_pipeline.normalize_rows(rows)

    products = run_stage("normalize_and_load_postgres", normalize_and_load_action, items=len(rows))
    print(f"normalized {len(products):,} products")
    id_to_index = {product.source_id: index for index, product in enumerate(products)}

    retrieval_env = stage_env_fingerprint(
        [
            "CARTSY_EMBEDDING_PROVIDER",
            "CARTSY_EMBEDDING_MODEL",
            "OPENAI_EMBEDDING_MODEL",
            "CARTSY_EMBEDDING_BATCH_SIZE",
            "CARTSY_FTS_CANDIDATES",
            "CARTSY_TRIGRAM_CANDIDATES",
            "CARTSY_TRIGRAM_MIN_SIMILARITY",
            "CARTSY_VECTOR_CANDIDATES",
            "CARTSY_VECTOR_REQUIRE_NEIGHBOR",
            "CARTSY_VECTOR_MIN_FTS_RANK",
            "CARTSY_VECTOR_MIN_TRIGRAM_SIMILARITY",
            "CARTSY_VECTOR_INCLUDE_NEIGHBORS",
            "CARTSY_EMBEDDING_DIMENSIONS",
        ]
    )
    retrieval_code = code_fingerprint(
        "pipeline.py",
        "embeddings.py",
        "scoring.py",
        "normalize.py",
        "utils/pipeline_helpers.py",
        "utils/pipeline_sql.py",
    )
    retrieval_key = retrieval_cache_key(
        normalization_key=cache_key,
        config=config,
        env=retrieval_env,
        code=retrieval_code,
    )
    retrieval_path = cache_path_for("retrieve_candidates", retrieval_key)
    stage_cache_status["retrieve_candidates"] = {"used": 0, "path": str(retrieval_path), "key": retrieval_key}

    print("retrieving candidate pairs")
    dedupe_pipeline.dev_log("stage start: retrieve_candidates")

    def retrieve_candidates_action():
        pair_blocks, blocking_stats = dedupe_pipeline.generate_candidate_pairs(
            products,
            config=config,
            normalization_key=cache_key,
        )
        stage_cache_status["retrieve_candidates"]["layers"] = dedupe_pipeline.retrieval_layer_cache_status
        return pair_blocks, blocking_stats

    pair_blocks, blocking_stats = run_stage("retrieve_candidates", retrieve_candidates_action, items=len(products))
    print(f"generated {len(pair_blocks):,} candidate pairs")

    scoring_code = code_fingerprint("pipeline.py", "embeddings.py", "scoring.py", "utils/pipeline_helpers.py", "utils/pipeline_sql.py")
    scoring_key = scoring_cache_key(
        retrieval_key=retrieval_key,
        config=config,
        code=scoring_code,
    )
    scoring_path = cache_path_for("score_candidates", scoring_key)
    stage_cache_status["score_candidates"] = {"used": 0, "path": str(scoring_path), "key": scoring_key}

    print("scoring candidate pairs")
    dedupe_pipeline.dev_log("stage start: score_candidates")

    def score_candidates_action():
        return dedupe_pipeline.score_candidate_pairs(
            products,
            pair_blocks,
            config=config,
            normalization_key=cache_key,
        )

    candidate_pairs, scored_candidate_pairs = run_stage("score_candidates", score_candidates_action, items=len(pair_blocks))

    clustering_code = code_fingerprint("pipeline.py", "clustering.py")
    cluster_key = clustering_cache_key(scoring_key=scoring_key, code=clustering_code)
    cluster_path = cache_path_for("cluster", cluster_key)
    stage_cache_status["cluster"] = {"used": 0, "path": str(cluster_path), "key": cluster_key}

    def cluster_action():
        dedupe_pipeline.dev_log("stage start: cluster")
        clusters, cluster_stats = dedupe_pipeline.build_clusters(products, candidate_pairs, id_to_index)
        return clusters, cluster_stats, invert_clusters(clusters)

    clusters, cluster_stats, source_to_cluster = run_stage("cluster", cluster_action, items=len(candidate_pairs))
    elapsed_seconds = perf_counter() - started
    report = dedupe_pipeline.build_summary_report(
        products=products,
        candidate_pairs=candidate_pairs,
        clusters=clusters,
        blocking_stats=blocking_stats,
        cluster_stats=cluster_stats,
        scored_candidate_pairs=scored_candidate_pairs,
        elapsed_seconds=elapsed_seconds,
    )
    report["run_id"] = output_path.name
    report["run_output_dir"] = str(output_path)
    report["normalization_cache"] = {
        "path": str(cache_path),
        "normalize_hash": normalize_hash,
    }
    report["stage_caches"] = stage_cache_status
    report["stage_timeline"] = stage_timeline
    report["run_timestamps"] = {
        "started_at_utc": run_started_at.isoformat(),
    }
    report["metrics"] = dedupe_pipeline.metrics.as_report(
        embedding_provider=dedupe_pipeline.embedding_provider,
        embedding_model=dedupe_pipeline.embedding_model,
        input_records=len(products),
        total_elapsed_seconds=elapsed_seconds,
    )

    def write_outputs_action():
        dedupe_pipeline.dev_log("stage start: write_outputs")
        write_outputs(
            output_path=output_path,
            products=products,
            candidate_pairs=candidate_pairs,
            clusters=clusters,
            source_to_cluster=source_to_cluster,
            report=report,
            near_miss_limit=config.near_miss_limit,
            sample_pair_limit=config.sample_pair_limit,
        )

    run_stage("write_outputs", write_outputs_action, items=len(products))
    elapsed_seconds = perf_counter() - started
    report["elapsed_seconds"] = round(elapsed_seconds, 3)
    run_ended_at = datetime.now(timezone.utc)
    report["run_timestamps"]["ended_at_utc"] = run_ended_at.isoformat()
    report["run_timestamps"]["elapsed_seconds"] = round(elapsed_seconds, 3)
    report["metrics"] = dedupe_pipeline.metrics.as_report(
        embedding_provider=dedupe_pipeline.embedding_provider,
        embedding_model=dedupe_pipeline.embedding_model,
        input_records=len(products),
        total_elapsed_seconds=elapsed_seconds,
    )
    (output_path / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DedupePipeline",
    "RunMetrics",
    "embedding_text",
    "postgres_retrieval_features",
    "run_pipeline",
]
