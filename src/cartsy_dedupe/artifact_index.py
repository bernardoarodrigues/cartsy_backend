from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider, configured_embedding_dimensions
from .query import make_query_embedding, register_pgvector
from .text import normalize_text
from .utils.pipeline_helpers import batched


@dataclass(frozen=True)
class ArtifactDocument:
    artifact_id: str
    artifact_type: str
    title: str
    search_text: str
    metadata: dict[str, Any]


def build_artifact_documents(run_dir: str | Path) -> list[ArtifactDocument]:
    run_path = Path(run_dir)
    assignments = read_csv_rows(run_path / "product_assignments.csv")
    assignments_by_source = {row.get("source_id", ""): row for row in assignments}
    docs: list[ArtifactDocument] = []
    docs.extend(group_documents(run_path, assignments_by_source))
    docs.extend(offer_documents(assignments))
    docs.extend(pair_documents(run_path, assignments_by_source))
    summary = summary_document(run_path)
    if summary is not None:
        docs.append(summary)
    deduped = {doc.artifact_id: doc for doc in docs}
    return list(deduped.values())


def index_artifacts(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    batch_size: int = 128,
    embed: bool = True,
) -> dict[str, object]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dependency is installed in the project venv.
        raise RuntimeError("Install psycopg[binary] before indexing artifacts.") from exc

    docs = build_artifact_documents(run_dir)
    resolved_run_id = run_id or Path(run_dir).resolve().name
    database_url = os.getenv("DATABASE_URL", "postgresql://cartsy:cartsy@localhost:5432/cartsy_matcher")
    embedding_dimensions = configured_embedding_dimensions()
    with psycopg.connect(database_url) as conn:
        register_pgvector(conn)
        ensure_artifact_index(conn, embedding_dimensions=embedding_dimensions)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cartsy_artifact_index WHERE run_id = %s", (resolved_run_id,))
            rows = [
                (
                    resolved_run_id,
                    doc.artifact_id,
                    doc.artifact_type,
                    doc.title,
                    doc.search_text,
                    json.dumps(doc.metadata, ensure_ascii=False),
                )
                for doc in docs
            ]
            cur.executemany(
                """
                INSERT INTO cartsy_artifact_index (
                    run_id, artifact_id, artifact_type, title, search_text, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                rows,
            )
            cur.execute(
                """
                UPDATE cartsy_artifact_index
                SET search_vector =
                    setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(search_text, '')), 'B')
                WHERE run_id = %s
                """,
                (resolved_run_id,),
            )
        conn.commit()
        embedded = 0
        if embed:
            embedded = embed_artifacts(conn, resolved_run_id, docs, batch_size=batch_size)
    return {"run_id": resolved_run_id, "artifact_count": len(docs), "embeddings_created": embedded}


def search_artifacts(
    query: str,
    *,
    run_id: str | None = None,
    limit: int = 10,
    artifact_type: str | None = None,
) -> list[dict[str, object]]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dependency is installed in the project venv.
        raise RuntimeError("Install psycopg[binary] before searching artifacts.") from exc

    query_norm = normalize_text(query)
    if not query_norm:
        return []
    database_url = os.getenv("DATABASE_URL", "postgresql://cartsy:cartsy@localhost:5432/cartsy_matcher")
    with psycopg.connect(database_url) as conn:
        register_pgvector(conn)
        query_embedding = make_query_embedding(query)
        with conn.cursor() as cur:
            params: dict[str, object] = {
                "query": query_norm,
                "run_id": run_id,
                "artifact_type": artifact_type,
                "limit": limit,
            }
            if query_embedding is not None:
                params["embedding"] = query_embedding
            cur.execute(artifact_search_sql(include_vector=query_embedding is not None), params)
            rows = cur.fetchall()
    return [
        {
            "score": round(float(score or 0.0), 4),
            "run_id": row_run_id,
            "artifact_id": artifact_id,
            "artifact_type": artifact_type_value,
            "title": title,
            "metadata": metadata,
            "retrieval_evidence": [item for item in evidence if item],
        }
        for row_run_id, artifact_id, artifact_type_value, title, metadata, score, evidence in rows
    ]


def ensure_artifact_index(conn: object, *, embedding_dimensions: int) -> None:
    vector_type = f"vector({embedding_dimensions})"
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS cartsy_artifact_index (
                run_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                title TEXT NOT NULL,
                search_text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding {vector_type},
                search_vector TSVECTOR,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (run_id, artifact_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cartsy_artifacts_type ON cartsy_artifact_index (run_id, artifact_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cartsy_artifacts_search ON cartsy_artifact_index USING GIN (search_vector)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cartsy_artifacts_title_trgm ON cartsy_artifact_index USING GIN (title gin_trgm_ops)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cartsy_artifacts_metadata ON cartsy_artifact_index USING GIN (metadata)")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cartsy_artifacts_embedding
            ON cartsy_artifact_index USING hnsw (embedding vector_cosine_ops)
            WHERE embedding IS NOT NULL
            """
        )
    conn.commit()


def embed_artifacts(conn: object, run_id: str, docs: list[ArtifactDocument], *, batch_size: int) -> int:
    embedder = EmbeddingProvider()
    embedded = 0
    for batch in batched(docs, batch_size):
        texts = [embedding_text_for_artifact(doc) for doc in batch]
        result = embedder.embed_texts(texts)
        updates = [(embedding, run_id, doc.artifact_id) for embedding, doc in zip(result.embeddings, batch, strict=True)]
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE cartsy_artifact_index SET embedding = %s WHERE run_id = %s AND artifact_id = %s",
                updates,
            )
        conn.commit()
        embedded += len(updates)
    return embedded


def artifact_search_sql(*, include_vector: bool) -> str:
    vector_select = (
        ", CASE WHEN embedding IS NOT NULL THEN 1 - (embedding <=> %(embedding)s) ELSE 0.0 END AS vector_score"
        if include_vector
        else ", 0.0::double precision AS vector_score"
    )
    vector_filter = "OR embedding IS NOT NULL" if include_vector else ""
    vector_evidence = (
        "CASE WHEN vector_score >= 0.78 THEN 'vector:cosine:' || round(vector_score::numeric, 4)::text END,"
        if include_vector
        else ""
    )
    return f"""
        WITH scored AS (
            SELECT run_id,
                   artifact_id,
                   artifact_type,
                   title,
                   metadata,
                   CASE
                       WHEN search_text = %(query)s THEN 1.0
                       WHEN position(%(query)s in search_text) > 0 THEN 0.99
                       ELSE 0.0
                   END AS exact_score,
                   CASE
                       WHEN search_vector @@ plainto_tsquery('simple', %(query)s)
                       THEN LEAST(1.0, ts_rank_cd(search_vector, plainto_tsquery('simple', %(query)s)) * 1.4)
                       ELSE 0.0
                   END AS fts_score,
                   GREATEST(similarity(title, %(query)s), similarity(search_text, %(query)s)) AS trigram_score
                   {vector_select}
            FROM cartsy_artifact_index
            WHERE (%(run_id)s IS NULL OR run_id = %(run_id)s)
              AND (%(artifact_type)s IS NULL OR artifact_type = %(artifact_type)s)
              AND (
                  title % %(query)s
                  OR search_text % %(query)s
                  OR search_vector @@ plainto_tsquery('simple', %(query)s)
                  OR position(%(query)s in search_text) > 0
                  {vector_filter}
              )
        )
        SELECT run_id,
               artifact_id,
               artifact_type,
               title,
               metadata,
               GREATEST(exact_score, fts_score, trigram_score, vector_score) AS score,
               ARRAY_REMOVE(ARRAY[
                   CASE WHEN exact_score >= 0.99 THEN 'exact:artifact_text' END,
                   CASE WHEN fts_score > 0 THEN 'lexical:fts:' || round(fts_score::numeric, 4)::text END,
                   CASE WHEN trigram_score >= 0.30 THEN 'trigram:title:' || round(trigram_score::numeric, 4)::text END,
                   {vector_evidence}
                   'backend:artifact_index'
               ], NULL) AS evidence
        FROM scored
        ORDER BY score DESC, title ASC
        LIMIT %(limit)s
    """


def group_documents(run_path: Path, assignments_by_source: dict[str, dict[str, str]]) -> list[ArtifactDocument]:
    docs: list[ArtifactDocument] = []
    for group in read_jsonl(run_path / "dedupe_groups.jsonl"):
        source_ids = [str(item) for item in group.get("source_ids", [])]
        offers = [assignments_by_source[source_id] for source_id in source_ids if source_id in assignments_by_source]
        offer_text = " ".join(format_offer(offer) for offer in offers[:20])
        title = str(group.get("canonical_name", ""))
        search_text = " ".join(
            part
            for part in [
                "group",
                str(group.get("dedupe_id", "")),
                title,
                str(group.get("canonical_brand", "")),
                str(group.get("canonical_category", "")),
                " ".join(str(item) for item in group.get("retailers", [])),
                " ".join(str(item) for item in group.get("merge_reasons", [])),
                offer_text,
            ]
            if part
        )
        metadata = {
            "dedupe_id": group.get("dedupe_id", ""),
            "source_ids": source_ids,
            "num_offers": group.get("num_offers", 0),
            "retailers": group.get("retailers", []),
            "canonical_brand": group.get("canonical_brand", ""),
            "canonical_category": group.get("canonical_category", ""),
            "cluster_confidence": group.get("cluster_confidence", ""),
            "price_min_cents": group.get("price_min_cents"),
            "price_max_cents": group.get("price_max_cents"),
            "graph": {"offers": [f"offer:{source_id}" for source_id in source_ids]},
        }
        docs.append(ArtifactDocument(f"group:{group.get('dedupe_id', '')}", "group", title, normalize_text(search_text), metadata))
    return docs


def offer_documents(assignments: list[dict[str, str]]) -> list[ArtifactDocument]:
    docs = []
    for row in assignments:
        title = row.get("name_raw", "")
        search_text = " ".join(
            [
                "offer",
                row.get("source_id", ""),
                row.get("dedupe_id", ""),
                row.get("retailer", ""),
                row.get("brand_raw", ""),
                row.get("sku", ""),
                row.get("dimension", ""),
                row.get("canonical_name", ""),
                row.get("canonical_brand", ""),
                row.get("decision", ""),
                row.get("explanation", ""),
                title,
            ]
        )
        metadata = {
            "source_id": row.get("source_id", ""),
            "dedupe_id": row.get("dedupe_id", ""),
            "retailer": row.get("retailer", ""),
            "brand": row.get("brand_raw", ""),
            "price_cents": row.get("price_cents", ""),
            "sku": row.get("sku", ""),
            "dimension": row.get("dimension", ""),
            "decision": row.get("decision", ""),
            "graph": {"group": f"group:{row.get('dedupe_id', '')}"},
        }
        docs.append(ArtifactDocument(f"offer:{row.get('source_id', '')}", "offer", title, normalize_text(search_text), metadata))
    return docs


def pair_documents(run_path: Path, assignments_by_source: dict[str, dict[str, str]]) -> list[ArtifactDocument]:
    docs: dict[str, ArtifactDocument] = {}
    for row in read_pair_rows(run_path):
        left_id = row.get("product_a_id", "")
        right_id = row.get("product_b_id", "")
        if not left_id or not right_id:
            continue
        left = assignments_by_source.get(left_id, {})
        right = assignments_by_source.get(right_id, {})
        title = f"{left.get('name_raw') or row.get('name_a', left_id)} <> {right.get('name_raw') or row.get('name_b', right_id)}"
        score = row.get("score", "")
        decision = row.get("decision", "")
        explanation = row.get("explanation", "")
        artifact_type = "near_miss" if decision == "no_merge" else "pair"
        search_text = " ".join(
            [
                artifact_type,
                left_id,
                right_id,
                score,
                decision,
                explanation,
                row.get("blocking_keys", ""),
                row.get("feature_scores", ""),
                format_offer(left),
                format_offer(right),
                row.get("name_a", ""),
                row.get("name_b", ""),
                row.get("brand_a", ""),
                row.get("brand_b", ""),
            ]
        )
        pair_key = ":".join(sorted([left_id, right_id]))
        metadata = {
            "product_a_id": left_id,
            "product_b_id": right_id,
            "dedupe_id_a": left.get("dedupe_id", ""),
            "dedupe_id_b": right.get("dedupe_id", ""),
            "score": score,
            "decision": decision,
            "explanation": explanation,
            "graph": {
                "offers": [f"offer:{left_id}", f"offer:{right_id}"],
                "groups": [f"group:{left.get('dedupe_id', '')}", f"group:{right.get('dedupe_id', '')}"],
            },
        }
        docs[f"{artifact_type}:{pair_key}"] = ArtifactDocument(
            f"{artifact_type}:{pair_key}",
            artifact_type,
            title,
            normalize_text(search_text),
            metadata,
        )
    return list(docs.values())


def summary_document(run_path: Path) -> ArtifactDocument | None:
    path = run_path / "summary_report.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    search_text = " ".join(["run summary", json.dumps(summary, ensure_ascii=False, sort_keys=True)])
    return ArtifactDocument("summary:run", "summary", f"Run summary for {run_path.name}", normalize_text(search_text), summary)


def read_pair_rows(run_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    parquet_path = run_path / "candidate_pairs.parquet"
    if parquet_path.exists():
        try:
            import polars as pl

            rows.extend({key: "" if value is None else str(value) for key, value in row.items()} for row in pl.read_parquet(parquet_path).to_dicts())
        except ImportError:
            pass
    rows.extend(read_csv_rows(run_path / "candidate_pairs.csv"))
    rows.extend(read_csv_rows(run_path / "near_miss_pairs.csv"))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_offer(row: dict[str, str]) -> str:
    return " ".join(
        part
        for part in [
            row.get("source_id", ""),
            row.get("retailer", ""),
            row.get("brand_raw", ""),
            row.get("name_raw", ""),
            row.get("sku", ""),
            row.get("dimension", ""),
            row.get("price_cents", ""),
        ]
        if part
    )


def embedding_text_for_artifact(doc: ArtifactDocument) -> str:
    return "\n".join(
        [
            f"type: {doc.artifact_type}",
            f"title: {doc.title}",
            f"text: {doc.search_text[:4000]}",
            f"metadata: {json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True)[:2000]}",
        ]
    )

