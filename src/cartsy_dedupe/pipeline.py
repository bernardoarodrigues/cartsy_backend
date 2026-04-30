from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.clustering import build_clusters
from cartsy_dedupe.ingest import load_rows
from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.reporting import build_summary_report
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct
from cartsy_dedupe.scoring import score_pair
from cartsy_dedupe.storage import prepare_output_dir, write_outputs
from cartsy_dedupe.text import informative_tokens, normalize_text

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

USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "text-embedding-3-small": {"input": 0.02},
}


@dataclass
class StageMetric:
    elapsed_seconds: float = 0.0
    items: int = 0

    def as_report(self) -> dict[str, float | int | None]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "items": self.items,
            "avg_seconds_per_item": round(self.elapsed_seconds / self.items, 6) if self.items else None,
        }


@dataclass
class UsageAccumulator:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.calls += 1
        input_tokens = usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = usage_value(usage, "total_tokens")
        cached_tokens = usage_nested_value(usage, "input_tokens_details", "cached_tokens")
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens or input_tokens + output_tokens

    def cost_usd(self, model: str) -> float:
        prices = USD_PER_1M_TOKENS.get(model, {})
        billable_input = max(0, self.input_tokens - self.cached_input_tokens)
        return (
            billable_input * prices.get("input", 0.0)
            + self.cached_input_tokens * prices.get("cached_input", prices.get("input", 0.0))
            + self.output_tokens * prices.get("output", 0.0)
        ) / 1_000_000

    def as_report(self, model: str) -> dict[str, float | int | str | None]:
        return {
            "model": model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_usd(model), 6),
            "pricing_note": "Estimated with standard OpenAI prices per 1M tokens configured in pipeline.py.",
        }


@dataclass
class RunMetrics:
    stages: dict[str, StageMetric] = field(default_factory=dict)
    openai_usage: dict[str, UsageAccumulator] = field(default_factory=lambda: defaultdict(UsageAccumulator))

    @contextmanager
    def stage(self, name: str, *, items: int = 0):
        started = perf_counter()
        try:
            yield
        finally:
            metric = self.stages.setdefault(name, StageMetric())
            metric.elapsed_seconds += perf_counter() - started
            metric.items += items

    def add_usage(self, model: str, usage: Any) -> None:
        self.openai_usage[model].add(usage)

    def as_report(self, *, embedding_model: str, extraction_model: str, input_records: int, total_elapsed_seconds: float) -> dict[str, object]:
        usage_by_model = {
            model: usage.as_report(model)
            for model, usage in sorted(self.openai_usage.items())
        }
        total_cost = sum(usage.cost_usd(model) for model, usage in self.openai_usage.items())
        return {
            "timing": {
                "total_elapsed_seconds": round(total_elapsed_seconds, 3),
                "input_records": input_records,
                "avg_seconds_per_input_record": round(total_elapsed_seconds / input_records, 6) if input_records else None,
                "stages": {name: metric.as_report() for name, metric in self.stages.items()},
            },
            "openai": {
                "embedding_model": embedding_model,
                "extraction_model": extraction_model,
                "usage_by_model": usage_by_model,
                "total_estimated_cost_usd": round(total_cost, 6),
                "cost_source": "OpenAI standard pricing checked 2026-04-30; update USD_PER_1M_TOKENS if model prices change.",
            },
        }


class ExtractedAttributes(BaseModel):
    brand: str | None = None
    product_line: str | None = None
    product_type: str | None = None
    category: str | None = None
    color: str | None = None
    size: str | None = None
    scent: str | None = None
    flavor: str | None = None
    material: str | None = None
    pack_count: str | None = None
    variant_name: str | None = None
    model_number: str | None = None
    sku_like_identifiers: list[str] = Field(default_factory=list)
    open_attributes: dict[str, str] = Field(default_factory=dict)


class PostgresOpenAIPipeline:
    """Postgres + pgvector + OpenAI implementation of the architecture doc."""

    name = "postgres_openai"

    def __init__(self) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        self.database_url = os.getenv("DATABASE_URL", "postgresql://cartsy:cartsy@localhost:5432/cartsy_matcher")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.extraction_model = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-5.4-nano")
        self.embedding_batch_size = int(os.getenv("CARTSY_EMBEDDING_BATCH_SIZE", "128"))
        self.llm_extraction_limit = int(os.getenv("CARTSY_LLM_EXTRACTION_LIMIT", "100"))
        self.fts_candidates = int(os.getenv("CARTSY_FTS_CANDIDATES", "25"))
        self.trigram_candidates = int(os.getenv("CARTSY_TRIGRAM_CANDIDATES", "25"))
        self.vector_candidates = int(os.getenv("CARTSY_VECTOR_CANDIDATES", "25"))
        self.embedding_dimensions = int(os.getenv("CARTSY_EMBEDDING_DIMENSIONS", "1536"))
        self.extracted_by_source_id: dict[str, dict[str, Any]] = {}
        self.embedding_count = 0
        self.extraction_count = 0
        self.metrics = RunMetrics()

    def normalize_rows(self, rows: Iterable[dict[str, str]]) -> list[NormalizedProduct]:
        products: list[NormalizedProduct] = []
        for idx, row in enumerate(rows, start=1):
            products.append(normalize_row(row))
            if idx % 50_000 == 0:
                print(f"normalized {idx:,} rows")

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
        with self.connect() as conn:
            pair_blocks, layer_counts = self.retrieve_candidate_pairs(conn, config)
            self.extract_candidate_attributes(conn, pair_blocks)
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
        for pair_number, ((left_index, right_index), block_keys) in enumerate(pair_blocks.items(), start=1):
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

        for source_id, brand, title, category, description, specs in rows:
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

        for batch in batched(rows, self.embedding_batch_size):
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
        self.add_candidate_rows(conn, pairs, counts, "exact", exact_candidate_sql(), (), config.max_candidate_pairs)
        exact_resolved_indexes = {
            index
            for pair, evidence in pairs.items()
            if any(key.startswith("exact:") for key in evidence)
            for index in pair
        }
        self.add_candidate_rows(
            conn,
            pairs,
            counts,
            "lexical",
            lexical_candidate_sql(),
            (self.fts_candidates,),
            config.max_candidate_pairs,
        )
        self.add_candidate_rows(
            conn,
            pairs,
            counts,
            "trigram",
            trigram_candidate_sql(),
            (self.trigram_candidates,),
            config.max_candidate_pairs,
        )
        cap_reached = config.max_candidate_pairs is not None and len(pairs) >= config.max_candidate_pairs
        if self.vector_candidates > 0 and not cap_reached:
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
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for left, right, evidence in cur:
                if left == right:
                    continue
                if left > right:
                    left, right = right, left
                pairs[(left, right)].add(str(evidence))
                counts[layer] += 1
                if max_candidate_pairs is not None and len(pairs) >= max_candidate_pairs:
                    return

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


def exact_keys(product: NormalizedProduct) -> dict[str, str]:
    keys: dict[str, str] = {}
    for key in ("ean", "gtin", "upc", "asin"):
        value = product.identifiers.get(key)
        if value:
            keys[key] = value
    if product.retailer and product.identifiers.get("sku"):
        keys[f"retailer_sku:{product.retailer}"] = product.identifiers["sku"]
    url_key = canonicalize_url(product.url)
    if url_key:
        keys["canonical_url"] = url_key
    return keys


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/").lower()
    if not host or not path:
        return ""
    return normalize_text(f"{host} {path}").replace(" ", "/")[:240]


def product_search_text(product: NormalizedProduct) -> str:
    tokens = informative_tokens(product.name_norm, limit=8)
    return " ".join(
        part
        for part in [
            product.brand_norm,
            " ".join(tokens),
            product.category_leaf,
            product.dimension_raw,
        ]
        if part
    )


def embedding_text(**parts: str | None) -> str:
    return "\n".join(f"{key}: {value}" for key, value in parts.items() if value)


def ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env or the environment before running the postgres_openai pipeline.")


def batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for index in range(0, len(items), max(1, size)):
        yield items[index : index + size]


def usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def usage_nested_value(usage: Any, parent_name: str, child_name: str) -> int:
    parent = usage.get(parent_name) if isinstance(usage, dict) else getattr(usage, parent_name, None)
    if parent is None:
        return 0
    value = parent.get(child_name) if isinstance(parent, dict) else getattr(parent, child_name, None)
    return int(value or 0)


def exact_candidate_sql() -> str:
    return """
        SELECT LEAST(a.product_index, b.product_index) AS left_index,
               GREATEST(a.product_index, b.product_index) AS right_index,
               'exact:' || a.key_type || ':' || left(a.key_value, 80) AS evidence
        FROM cartsy_exact_keys a
        JOIN cartsy_exact_keys b
          ON a.key_type = b.key_type
         AND a.key_value = b.key_value
         AND a.product_index < b.product_index
    """


def lexical_candidate_sql() -> str:
    return """
        SELECT p.source_index, q.source_index,
               'lexical:fts:' || round(q.rank::numeric, 4)::text AS evidence
        FROM cartsy_products p
        JOIN LATERAL (
            SELECT candidate.source_index,
                   ts_rank_cd(candidate.search_vector, plainto_tsquery('simple', p.search_text)) AS rank
            FROM cartsy_products candidate
            WHERE candidate.source_index > p.source_index
              AND p.search_text <> ''
              AND candidate.brand_norm = p.brand_norm
              AND candidate.brand_norm <> ''
              AND candidate.search_vector @@ plainto_tsquery('simple', p.search_text)
            ORDER BY rank DESC
            LIMIT %s
        ) q ON true
    """


def trigram_candidate_sql() -> str:
    return """
        SELECT p.source_index, q.source_index,
               'trigram:title:' || round(q.similarity::numeric, 4)::text AS evidence
        FROM cartsy_products p
        JOIN LATERAL (
            SELECT candidate.source_index,
                   similarity(candidate.name_norm, p.name_norm) AS similarity
            FROM cartsy_products candidate
            WHERE candidate.source_index > p.source_index
              AND candidate.brand_norm = p.brand_norm
              AND candidate.brand_norm <> ''
              AND candidate.name_norm % p.name_norm
            ORDER BY similarity DESC
            LIMIT %s
        ) q ON true
        WHERE q.similarity >= 0.45
    """


def vector_candidate_sql() -> str:
    return """
        SELECT p.source_index, q.source_index,
               'vector:cosine:' || round(q.similarity::numeric, 4)::text AS evidence
        FROM cartsy_products p
        JOIN LATERAL (
            SELECT candidate.source_index,
                   1 - (candidate.embedding <=> p.embedding) AS similarity
            FROM cartsy_products candidate
            WHERE p.embedding IS NOT NULL
              AND candidate.embedding IS NOT NULL
              AND candidate.source_index > p.source_index
            ORDER BY candidate.embedding <=> p.embedding
            LIMIT %s
        ) q ON true
        WHERE q.similarity >= 0.78
    """


def postgres_retrieval_features(block_keys: set[str]) -> dict[str, float]:
    features = {"exact": 0.0, "lexical": 0.0, "trigram": 0.0, "vector": 0.0}
    for key in block_keys:
        if key.startswith("exact:"):
            features["exact"] = max(features["exact"], 1.0)
        elif key.startswith("lexical:fts:"):
            features["lexical"] = max(features["lexical"], evidence_value(key, default=0.70))
        elif key.startswith("trigram:title:"):
            features["trigram"] = max(features["trigram"], evidence_value(key, default=0.45))
        elif key.startswith("vector:cosine:"):
            features["vector"] = max(features["vector"], evidence_value(key, default=0.78))
    features["lexical"] = min(1.0, features["lexical"] * 1.4)
    features["trigram"] = min(1.0, features["trigram"])
    features["vector"] = min(1.0, features["vector"])
    return features


def evidence_value(key: str, *, default: float) -> float:
    try:
        return float(key.rsplit(":", 1)[1])
    except ValueError:
        return default


def extracted_attribute_score(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[float, str, list[str]]:
    if not left or not right:
        return 0.45, "unknown", []
    positive = 0
    comparable = 0
    conflicts: list[str] = []
    reasons: list[str] = []
    for key in (
        "brand",
        "product_line",
        "product_type",
        "category",
        "variant_name",
        "color",
        "size",
        "scent",
        "flavor",
        "material",
        "pack_count",
        "model_number",
    ):
        left_value = normalize_text(left.get(key))
        right_value = normalize_text(right.get(key))
        if not left_value or not right_value:
            continue
        comparable += 1
        if left_value == right_value:
            positive += 1
            reasons.append(f"llm_{key}_match:{left_value}")
        elif key in {"variant_name", "color", "size", "scent", "flavor", "material", "pack_count", "model_number"}:
            conflicts.append(f"llm_{key}_conflict")
    if comparable == 0:
        return 0.45, "unknown", reasons
    score = positive / comparable
    relation = "same_parent_different_variant" if conflicts and same_parent_attributes(left, right) else "unknown"
    reasons.extend(conflicts)
    return score, relation, reasons


def same_parent_attributes(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("brand", "product_line", "product_type"):
        left_value = normalize_text(left.get(key))
        right_value = normalize_text(right.get(key))
        if left_value and right_value and left_value != right_value:
            return False
    return bool(normalize_text(left.get("product_line")) and normalize_text(right.get("product_line")))


def run_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    limit: int | None = None,
) -> dict[str, object]:
    started = perf_counter()
    output_path = prepare_output_dir(output_dir)
    dedupe_pipeline = PostgresOpenAIPipeline()

    print(f"loading {input_path}")
    with dedupe_pipeline.metrics.stage("load_rows"):
        rows = load_rows(input_path, limit=limit)
    print(f"loaded {len(rows):,} rows")

    print("normalizing and loading Postgres")
    with dedupe_pipeline.metrics.stage("normalize_and_load_postgres", items=len(rows)):
        products = dedupe_pipeline.normalize_rows(rows)
    print(f"normalized {len(products):,} products")
    id_to_index = {product.source_id: index for index, product in enumerate(products)}

    print("retrieving candidate pairs")
    with dedupe_pipeline.metrics.stage("retrieve_candidates", items=len(products)):
        pair_blocks, blocking_stats = dedupe_pipeline.generate_candidate_pairs(products, config=config)
    print(f"generated {len(pair_blocks):,} candidate pairs")

    print("scoring candidate pairs")
    with dedupe_pipeline.metrics.stage("score_candidates", items=len(pair_blocks)):
        candidate_pairs, scored_candidate_pairs = dedupe_pipeline.score_candidate_pairs(products, pair_blocks, config=config)

    with dedupe_pipeline.metrics.stage("cluster", items=len(candidate_pairs)):
        clusters, cluster_stats = dedupe_pipeline.build_clusters(products, candidate_pairs, id_to_index)
        source_to_cluster = invert_clusters(clusters)
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
    report["metrics"] = dedupe_pipeline.metrics.as_report(
        embedding_model=dedupe_pipeline.embedding_model,
        extraction_model=dedupe_pipeline.extraction_model,
        input_records=len(products),
        total_elapsed_seconds=elapsed_seconds,
    )
    with dedupe_pipeline.metrics.stage("write_outputs", items=len(products)):
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
    elapsed_seconds = perf_counter() - started
    report["elapsed_seconds"] = round(elapsed_seconds, 3)
    report["metrics"] = dedupe_pipeline.metrics.as_report(
        embedding_model=dedupe_pipeline.embedding_model,
        extraction_model=dedupe_pipeline.extraction_model,
        input_records=len(products),
        total_elapsed_seconds=elapsed_seconds,
    )
    (output_path / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def invert_clusters(clusters: dict[str, dict[str, object]]) -> dict[str, str]:
    source_to_cluster: dict[str, str] = {}
    for dedupe_id, cluster in clusters.items():
        for source_id in cluster["source_ids"]:
            source_to_cluster[str(source_id)] = dedupe_id
    return source_to_cluster


__all__ = [
    "PostgresOpenAIPipeline",
    "ExtractedAttributes",
    "embedding_text",
    "postgres_retrieval_features",
    "run_pipeline",
]
