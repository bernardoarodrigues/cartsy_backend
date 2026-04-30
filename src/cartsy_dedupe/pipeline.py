from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from dotenv import load_dotenv
from tqdm import tqdm

from cartsy_dedupe.clustering import build_clusters
from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.ingest import load_rows
from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.reporting import build_summary_report
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct
from cartsy_dedupe.scoring import score_pair
from cartsy_dedupe.storage import prepare_output_dir, write_outputs
from cartsy_dedupe.utils.pipeline_cache import (
    cache_path_for,
    candidate_pairs_from_records,
    candidate_pairs_to_records,
    clustering_cache_key,
    code_fingerprint,
    normalization_cache_dir,
    normalization_cache_key,
    normalize_module_hash,
    pair_blocks_from_records,
    pair_blocks_to_records,
    product_signature,
    read_normalization_cache,
    read_stage_cache,
    retrieval_cache_key,
    scoring_cache_key,
    stage_env_fingerprint,
    write_normalization_cache,
    write_stage_cache,
)
from cartsy_dedupe.utils.pipeline_helpers import (
    ExtractedAttributes,
    batched,
    embedding_text,
    ensure_openai_api_key,
    exact_keys,
    extracted_attribute_score,
    invert_clusters,
    product_search_text,
)
from cartsy_dedupe.utils.pipeline_metrics import RunMetrics
from cartsy_dedupe.utils.pipeline_sql import (
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

try:  # pragma: no cover - external package availability is covered by smoke runs.
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

T = TypeVar("T")
PairBlocks = dict[tuple[int, int], set[str]]
Clusters = dict[str, dict[str, object]]


class DedupePipeline:
    """Postgres + pgvector + OpenAI implementation of the architecture doc."""

    name = "postgres_openai"

    def __init__(self, *, dev: bool = False) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        self.database_url = os.getenv("DATABASE_URL", "postgresql://cartsy:cartsy@localhost:5432/cartsy_matcher")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.extraction_model = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-5.4-nano")
        self.embedding_batch_size = int(os.getenv("CARTSY_EMBEDDING_BATCH_SIZE", "128"))
        self.llm_extraction_limit = int(os.getenv("CARTSY_LLM_EXTRACTION_LIMIT", "100"))
        self.fts_candidates = int(os.getenv("CARTSY_FTS_CANDIDATES", "25"))
        self.trigram_candidates = int(os.getenv("CARTSY_TRIGRAM_CANDIDATES", "25"))
        self.trigram_min_similarity = float(os.getenv("CARTSY_TRIGRAM_MIN_SIMILARITY", "0.55"))
        self.vector_candidates = int(os.getenv("CARTSY_VECTOR_CANDIDATES", "25"))
        self.embedding_dimensions = int(os.getenv("CARTSY_EMBEDDING_DIMENSIONS", "1536"))
        self.extracted_by_source_id: dict[str, dict[str, Any]] = {}
        self.embedding_count = 0
        self.extraction_count = 0
        self.metrics = RunMetrics()
        self.dev = dev

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
    ) -> tuple[PairBlocks, dict[str, int]]:
        self.dev_log("running retrieval stages: exact -> lexical -> trigram -> vector")
        with self.connect() as conn:
            pair_blocks, layer_counts = self.retrieve_candidate_pairs(conn, config)
            self.dev_log("running candidate attribute extraction")
            self.extract_candidate_attributes(conn, pair_blocks)
            self.dev_log("loading extracted attributes into normalized products")
            self.load_extracted_attributes(conn, products)
        stats = {
            "candidate_cap_reached": int(config.max_candidate_pairs is not None and len(pair_blocks) >= config.max_candidate_pairs),
            "exact_pairs": layer_counts.get("exact", 0),
            "lexical_pairs": layer_counts.get("lexical", 0),
            "trigram_pairs": layer_counts.get("trigram", 0),
            "vector_pairs": layer_counts.get("vector", 0),
            "blocking_keys": sum(layer_counts.values()),
            "skipped_blocks": 0,
            "oversized_block_rows": 0,
            "openai_embeddings_created": self.embedding_count,
            "openai_extractions_created": self.extraction_count,
        }
        return pair_blocks, stats

    def score_candidate_pairs(
        self,
        products: list[NormalizedProduct],
        pair_blocks: PairBlocks,
        *,
        config: PipelineConfig,
    ) -> tuple[list[CandidatePair], int]:
        candidate_pairs: list[CandidatePair] = []
        pair_items = pair_blocks.items()
        for pair_number, ((left_index, right_index), block_keys) in enumerate(
            self.progress(pair_items, total=len(pair_blocks), desc="score pairs", unit="pair"),
            start=1,
        ):
            left = products[left_index]
            right = products[right_index]
            pair = self.score_postgres_pair(left, right, block_keys, config)
            if pair.decision == "no_merge" and pair.score < config.near_miss_threshold:
                continue
            candidate_pairs.append(pair)
            if pair_number % 100_000 == 0:
                print(f"scored {pair_number:,} candidate pairs; kept {len(candidate_pairs):,}")
        return candidate_pairs, len(pair_blocks)

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
            "embedding_model": self.embedding_model,
            "extraction_model": self.extraction_model,
            "cascade": "exact keys -> full-text retrieval -> trigram retrieval -> vector retrieval -> explainable rerank",
        }
        return report

    def connect(self):
        if psycopg is None:
            raise RuntimeError("Install psycopg[binary] and pgvector before running the postgres_openai pipeline.")
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
            raise RuntimeError("Install pgvector before running the postgres_openai pipeline.")
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
                    extracted_attributes JSONB NOT NULL DEFAULT '{{}}'::jsonb,
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
            cur.execute("CREATE INDEX idx_cartsy_products_attrs ON cartsy_products USING GIN (extracted_attributes)")
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

    def extract_candidate_attributes(self, conn, pair_blocks: PairBlocks) -> None:
        if self.llm_extraction_limit <= 0:
            return
        candidate_indexes = sorted({index for pair in pair_blocks for index in pair})
        if not candidate_indexes:
            return
        if OpenAI is None:
            raise RuntimeError("Install openai before running LLM attribute extraction.")
        ensure_openai_api_key()
        client = OpenAI()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_id, brand_raw, name_raw, category_raw, description_raw, specs_raw
                FROM cartsy_products
                WHERE source_index = ANY(%s)
                  AND extracted_attributes = '{}'::jsonb
                ORDER BY source_index
                LIMIT %s
                """,
                (candidate_indexes, self.llm_extraction_limit),
            )
            rows = cur.fetchall()

        self.dev_log(f"extracting attributes for up to {len(rows):,} candidate products")
        for source_id, brand, title, category, description, specs in self.progress(
            rows,
            total=len(rows),
            desc="extract attrs",
            unit="product",
        ):
            attrs = self.extract_attributes_with_openai(client, brand, title, category, description, specs)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE cartsy_products SET extracted_attributes = %s::jsonb WHERE source_id = %s",
                    (json.dumps(attrs, ensure_ascii=False), source_id),
                )
            self.extraction_count += 1
        conn.commit()

    def extract_attributes_with_openai(
        self,
        client: Any,
        brand: str,
        title: str,
        category: str,
        description: str,
        specs: str,
    ) -> dict[str, Any]:
        prompt = (
            "Extract product matching attributes. Use null when unknown. "
            "Separate parent product line from variant details like variant name, size, color, scent, flavor, material, and pack count.\n\n"
            f"Brand: {brand}\nTitle: {title}\nCategory: {category}\nDescription: {description[:1500]}\nSpecs: {specs[:1500]}"
        )
        response = client.responses.parse(
            model=self.extraction_model,
            input=[
                {
                    "role": "system",
                    "content": "You extract structured retail product attributes for deduplication. Return only schema fields.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=ExtractedAttributes,
        )
        self.metrics.add_usage(self.extraction_model, getattr(response, "usage", None))
        parsed = response.output_parsed
        if parsed is None:
            return {}
        return parsed.model_dump(exclude_none=True)

    def embed_products(self, conn, *, exclude_indexes: set[int] | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError("Install openai before running embedding generation.")
        ensure_openai_api_key()
        client = OpenAI()
        exclude_indexes = exclude_indexes or set()
        with conn.cursor() as cur:
            if exclude_indexes:
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

        batches = list(batched(rows, self.embedding_batch_size))
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
            response = client.embeddings.create(model=self.embedding_model, input=texts)
            self.metrics.add_usage(self.embedding_model, getattr(response, "usage", None))
            updates = [(item.embedding, row[0]) for item, row in zip(response.data, batch, strict=True)]
            with conn.cursor() as cur:
                cur.executemany("UPDATE cartsy_products SET embedding = %s WHERE source_id = %s", updates)
            self.embedding_count += len(updates)
            print(f"embedded {self.embedding_count:,} products")
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cartsy_products_embedding
                ON cartsy_products USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL
                """
            )
        conn.commit()

    def load_extracted_attributes(self, conn, products: list[NormalizedProduct]) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT source_id, extracted_attributes FROM cartsy_products WHERE extracted_attributes <> '{}'::jsonb")
            self.extracted_by_source_id = {source_id: attrs for source_id, attrs in cur.fetchall()}
        products_by_id = {product.source_id: product for product in products}
        for source_id, attrs in self.extracted_by_source_id.items():
            product = products_by_id.get(source_id)
            if product is not None:
                product.extracted_attributes = attrs

    def retrieve_candidate_pairs(self, conn, config: PipelineConfig) -> tuple[PairBlocks, Counter[str]]:
        pairs: PairBlocks = defaultdict(set)
        counts: Counter[str] = Counter()
        self.dev_log("retrieval stage: exact keys")
        self.add_candidate_rows(conn, pairs, counts, "exact", exact_candidate_sql(), (), config.max_candidate_pairs)
        exact_resolved_indexes = {
            index
            for pair, evidence in pairs.items()
            if any(key.startswith("exact:") for key in evidence)
            for index in pair
        }
        self.dev_log("retrieval stage: lexical FTS")
        self.add_candidate_rows(
            conn,
            pairs,
            counts,
            "lexical",
            lexical_candidate_sql(),
            (self.fts_candidates,),
            config.max_candidate_pairs,
        )
        self.dev_log("retrieval stage: trigram")
        self.add_candidate_rows(
            conn,
            pairs,
            counts,
            "trigram",
            trigram_candidate_sql(),
            (self.trigram_min_similarity, self.trigram_candidates),
            config.max_candidate_pairs,
        )
        cap_reached = config.max_candidate_pairs is not None and len(pairs) >= config.max_candidate_pairs
        if self.vector_candidates > 0 and not cap_reached:
            self.dev_log("retrieval stage: vector embeddings")
            self.embed_products(conn, exclude_indexes=exact_resolved_indexes)
            self.add_candidate_rows(
                conn,
                pairs,
                counts,
                "vector",
                vector_candidate_sql(),
                (self.vector_candidates,),
                config.max_candidate_pairs,
            )
        return pairs, counts

    def add_candidate_rows(
        self,
        conn,
        pairs: PairBlocks,
        counts: Counter[str],
        layer: str,
        sql: str,
        params: tuple[object, ...],
        max_candidate_pairs: int | None,
    ) -> None:
        if max_candidate_pairs is not None and len(pairs) >= max_candidate_pairs:
            return
        show_progress = layer in {"lexical", "trigram", "vector"}
        cursor_name = f"cartsy_{layer}_{int(perf_counter() * 1_000_000)}"
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
                        pairs[(left, right)].add(str(evidence))
                        counts[layer] += 1
                        if max_candidate_pairs is not None and len(pairs) >= max_candidate_pairs:
                            return
            finally:
                if bar is not None:
                    bar.close()

    def score_postgres_pair(
        self,
        left: NormalizedProduct,
        right: NormalizedProduct,
        block_keys: set[str],
        config: PipelineConfig,
    ) -> CandidatePair:
        rule_result = score_pair(left, right, merge_threshold=config.merge_threshold)
        retrieval = postgres_retrieval_features(block_keys)
        attr_score, attr_relation, attr_reasons = extracted_attribute_score(
            self.extracted_by_source_id.get(left.source_id, {}),
            self.extracted_by_source_id.get(right.source_id, {}),
        )
        score = (
            0.54 * rule_result.score
            + 0.14 * retrieval["exact"]
            + 0.10 * retrieval["lexical"]
            + 0.08 * retrieval["trigram"]
            + 0.10 * retrieval["vector"]
            + 0.04 * attr_score
        )
        if retrieval["exact"] >= 1.0:
            score = max(score, 0.97)
        elif retrieval["vector"] >= 0.90 and retrieval["lexical"] >= 0.70 and not rule_result.auto_blocked:
            score = max(score, 0.86)

        relation = "exact_match"
        if rule_result.auto_blocked or attr_relation == "same_parent_different_variant":
            relation = "same_parent_different_variant"
            score = min(score, config.merge_threshold - 0.01)
        elif score < config.merge_threshold and score >= config.near_miss_threshold:
            relation = "similar_related_product"
        elif score < config.near_miss_threshold:
            relation = "no_match"

        score = max(0.0, min(1.0, score))
        decision = "merge" if relation == "exact_match" and score >= config.merge_threshold else "no_merge"
        explanations = [
            f"relation:{relation}",
            f"rule_score:{rule_result.score:.2f}",
            f"exact:{retrieval['exact']:.2f}",
            f"fts:{retrieval['lexical']:.2f}",
            f"trigram:{retrieval['trigram']:.2f}",
            f"vector:{retrieval['vector']:.2f}",
            f"llm_attrs:{attr_score:.2f}",
        ]
        if rule_result.explanation:
            explanations.append(rule_result.explanation)
        explanations.extend(attr_reasons[:4])
        feature_scores = {
            **rule_result.feature_scores,
            "postgres_exact": retrieval["exact"],
            "postgres_fts": retrieval["lexical"],
            "postgres_trigram": retrieval["trigram"],
            "postgres_vector": retrieval["vector"],
            "llm_attributes": attr_score,
        }
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
    normalization_signature = ""
    cache_used = False
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
        nonlocal cache_used
        cached_products = read_normalization_cache(cache_path)
        if cached_products is not None:
            products = cached_products
            cache_used = True
            print(f"loaded {len(products):,} normalized products from cache")
            with dedupe_pipeline.connect() as conn:
                dedupe_pipeline.reset_database(conn)
                dedupe_pipeline.insert_products(conn, products)
                dedupe_pipeline.insert_exact_keys(conn, products)
        else:
            products = dedupe_pipeline.normalize_rows(rows)
            write_normalization_cache(
                cache_path,
                products=products,
                metadata={
                    "cache_schema_version": 1,
                    "input_path": str(resolved_input_path),
                    "limit": limit,
                    "normalize_hash": normalize_hash,
                },
            )
            print(f"saved normalized cache: {cache_path}")
        return products

    products = run_stage("normalize_and_load_postgres", normalize_and_load_action, items=len(rows))
    normalization_signature = product_signature(products)
    print(f"normalized {len(products):,} products")
    id_to_index = {product.source_id: index for index, product in enumerate(products)}

    retrieval_env = stage_env_fingerprint(
        [
            "OPENAI_EMBEDDING_MODEL",
            "OPENAI_EXTRACTION_MODEL",
            "CARTSY_EMBEDDING_BATCH_SIZE",
            "CARTSY_LLM_EXTRACTION_LIMIT",
            "CARTSY_FTS_CANDIDATES",
            "CARTSY_TRIGRAM_CANDIDATES",
            "CARTSY_TRIGRAM_MIN_SIMILARITY",
            "CARTSY_VECTOR_CANDIDATES",
            "CARTSY_EMBEDDING_DIMENSIONS",
        ]
    )
    retrieval_code = code_fingerprint("pipeline.py", "scoring.py", "normalize.py", "utils/pipeline_helpers.py", "utils/pipeline_sql.py")
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
        cached = read_stage_cache(retrieval_path)
        if cached is not None:
            payload = cached["payload"]
            pair_blocks = pair_blocks_from_records(payload.get("pair_blocks") or [])
            blocking_stats = dict(payload.get("blocking_stats") or {})
            extracted_by_source_id = dict(payload.get("extracted_by_source_id") or {})
            dedupe_pipeline.extracted_by_source_id = extracted_by_source_id
            dedupe_pipeline.embedding_count = int(payload.get("embedding_count") or 0)
            dedupe_pipeline.extraction_count = int(payload.get("extraction_count") or 0)
            products_by_id = {product.source_id: product for product in products}
            for source_id, attrs in extracted_by_source_id.items():
                product = products_by_id.get(source_id)
                if product is not None:
                    product.extracted_attributes = dict(attrs or {})
            stage_cache_status["retrieve_candidates"]["used"] = 1
            return pair_blocks, blocking_stats

        pair_blocks, blocking_stats = dedupe_pipeline.generate_candidate_pairs(products, config=config)
        write_stage_cache(
            retrieval_path,
            metadata={
                "stage": "retrieve_candidates",
                "normalization_key": cache_key,
                "normalization_signature": normalization_signature,
                "config": asdict(config),
                "env": retrieval_env,
                "code": retrieval_code,
            },
            payload={
                "pair_blocks": pair_blocks_to_records(pair_blocks),
                "blocking_stats": blocking_stats,
                "extracted_by_source_id": dedupe_pipeline.extracted_by_source_id,
                "embedding_count": dedupe_pipeline.embedding_count,
                "extraction_count": dedupe_pipeline.extraction_count,
            },
        )
        return pair_blocks, blocking_stats

    pair_blocks, blocking_stats = run_stage("retrieve_candidates", retrieve_candidates_action, items=len(products))
    print(f"generated {len(pair_blocks):,} candidate pairs")

    scoring_code = code_fingerprint("pipeline.py", "scoring.py", "utils/pipeline_helpers.py", "utils/pipeline_sql.py")
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
        cached = read_stage_cache(scoring_path)
        if cached is not None:
            payload = cached["payload"]
            stage_cache_status["score_candidates"]["used"] = 1
            return (
                candidate_pairs_from_records(payload.get("candidate_pairs") or []),
                int(payload.get("scored_candidate_pairs") or 0),
            )

        candidate_pairs, scored_candidate_pairs = dedupe_pipeline.score_candidate_pairs(products, pair_blocks, config=config)
        write_stage_cache(
            scoring_path,
            metadata={
                "stage": "score_candidates",
                "retrieval_key": retrieval_key,
                "normalization_signature": normalization_signature,
                "config": asdict(config),
                "code": scoring_code,
            },
            payload={
                "candidate_pairs": candidate_pairs_to_records(candidate_pairs),
                "scored_candidate_pairs": scored_candidate_pairs,
            },
        )
        return candidate_pairs, scored_candidate_pairs

    candidate_pairs, scored_candidate_pairs = run_stage("score_candidates", score_candidates_action, items=len(pair_blocks))

    clustering_code = code_fingerprint("pipeline.py", "clustering.py")
    cluster_key = clustering_cache_key(scoring_key=scoring_key, code=clustering_code)
    cluster_path = cache_path_for("cluster", cluster_key)
    stage_cache_status["cluster"] = {"used": 0, "path": str(cluster_path), "key": cluster_key}

    def cluster_action():
        cached = read_stage_cache(cluster_path)
        if cached is not None:
            payload = cached["payload"]
            clusters = dict(payload.get("clusters") or {})
            cluster_stats = {
                str(key): int(value)
                for key, value in dict(payload.get("cluster_stats") or {}).items()
            }
            source_to_cluster = {
                str(key): str(value)
                for key, value in dict(payload.get("source_to_cluster") or {}).items()
            }
            stage_cache_status["cluster"]["used"] = 1
            return clusters, cluster_stats, source_to_cluster

        dedupe_pipeline.dev_log("stage start: cluster")
        clusters, cluster_stats = dedupe_pipeline.build_clusters(products, candidate_pairs, id_to_index)
        source_to_cluster = invert_clusters(clusters)
        write_stage_cache(
            cluster_path,
            metadata={
                "stage": "cluster",
                "scoring_key": scoring_key,
                "normalization_signature": normalization_signature,
                "code": clustering_code,
            },
            payload={
                "clusters": clusters,
                "cluster_stats": cluster_stats,
                "source_to_cluster": source_to_cluster,
            },
        )
        return clusters, cluster_stats, source_to_cluster

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
        "used": int(cache_used),
        "path": str(cache_path),
        "normalize_hash": normalize_hash,
    }
    report["stage_caches"] = stage_cache_status
    report["stage_timeline"] = stage_timeline
    report["run_timestamps"] = {
        "started_at_utc": run_started_at.isoformat(),
    }
    report["metrics"] = dedupe_pipeline.metrics.as_report(
        embedding_model=dedupe_pipeline.embedding_model,
        extraction_model=dedupe_pipeline.extraction_model,
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
        embedding_model=dedupe_pipeline.embedding_model,
        extraction_model=dedupe_pipeline.extraction_model,
        input_records=len(products),
        total_elapsed_seconds=elapsed_seconds,
    )
    (output_path / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DedupePipeline",
    "ExtractedAttributes",
    "RunMetrics",
    "embedding_text",
    "extracted_attribute_score",
    "postgres_retrieval_features",
    "run_pipeline",
]
