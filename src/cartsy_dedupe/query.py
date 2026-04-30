from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .text import normalize_text

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency is installed in the project venv.
    import difflib

    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FallbackFuzz()


def search_products(
    run_dir: str | Path,
    query: str,
    *,
    limit: int = 10,
    backend: str = "auto",
) -> list[dict[str, object]]:
    if backend not in {"auto", "artifacts", "postgres"}:
        raise ValueError("backend must be one of: auto, artifacts, postgres")
    if backend in {"auto", "postgres"}:
        try:
            postgres_results = search_products_postgres(run_dir, query, limit=limit)
        except RuntimeError:
            if backend == "postgres":
                raise
        else:
            if postgres_results or backend == "postgres":
                return postgres_results
    return search_products_artifacts(run_dir, query, limit=limit)


def search_products_artifacts(run_dir: str | Path, query: str, *, limit: int = 10) -> list[dict[str, object]]:
    query_norm = normalize_text(query)
    rows = read_assignments(run_dir)
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        haystack = " ".join(
            [
                row.get("name_raw", ""),
                row.get("brand_raw", ""),
                row.get("canonical_name", ""),
                row.get("canonical_brand", ""),
                row.get("retailer", ""),
                row.get("sku", ""),
                row.get("dimension", ""),
            ]
        )
        score = float(fuzz.token_set_ratio(query_norm, normalize_text(haystack))) / 100.0
        if query_norm in normalize_text(haystack):
            score = max(score, 0.99)
        scored.append((score, row))

    results = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [
        {
            "score": round(score, 4),
            "source_id": row.get("source_id", ""),
            "dedupe_id": row.get("dedupe_id", ""),
            "retailer": row.get("retailer", ""),
            "name": row.get("name_raw", ""),
            "brand": row.get("brand_raw", ""),
            "price_cents": row.get("price_cents", ""),
            "cluster_confidence": row.get("cluster_confidence", ""),
            "decision": row.get("decision", ""),
        }
        for score, row in results
    ]


def search_products_postgres(run_dir: str | Path, query: str, *, limit: int = 10) -> list[dict[str, object]]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dependency is installed in the project venv.
        raise RuntimeError("Install psycopg[binary] before using the postgres search backend.") from exc

    query_norm = normalize_text(query)
    if not query_norm:
        return []

    database_url = os.getenv("DATABASE_URL", "postgresql://cartsy:cartsy@localhost:5432/cartsy_matcher")
    assignments = {row["source_id"]: row for row in read_assignments(run_dir)}
    try:
        with psycopg.connect(database_url) as conn:
            register_pgvector(conn)
            query_embedding = make_query_embedding(query)
            with conn.cursor() as cur:
                params: list[object] = [query_norm] * 16
                if query_embedding is None:
                    cur.execute(postgres_search_sql(include_vector=False), (*params, limit))
                else:
                    cur.execute(postgres_search_sql(include_vector=True), (*params[:11], query_embedding, *params[11:], limit))
                rows = cur.fetchall()
    except Exception as exc:  # pragma: no cover - depends on local service state.
        raise RuntimeError("Could not search Postgres. Start `docker compose up -d postgres` or use --backend artifacts.") from exc

    results: list[dict[str, object]] = []
    for row in rows:
        source_id, retailer, brand, name, price_cents, score, evidence = row
        assignment = assignments.get(source_id, {})
        results.append(
            {
                "score": round(float(score or 0.0), 4),
                "source_id": source_id,
                "dedupe_id": assignment.get("dedupe_id", ""),
                "retailer": assignment.get("retailer", retailer),
                "name": assignment.get("name_raw", name),
                "brand": assignment.get("brand_raw", brand),
                "price_cents": assignment.get("price_cents", price_cents),
                "cluster_confidence": assignment.get("cluster_confidence", ""),
                "decision": assignment.get("decision", ""),
                "search_backend": "postgres",
                "retrieval_evidence": [item for item in evidence if item],
            }
        )
    return results


def make_query_embedding(query: str) -> list[float] | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - dependency is installed in the project venv.
        return None
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI()
    try:
        response = client.embeddings.create(model=model, input=[f"title: {query}"])
    except Exception:
        return None
    return list(response.data[0].embedding)


def register_pgvector(conn: object) -> None:
    try:
        from pgvector.psycopg import register_vector
    except ImportError:  # pragma: no cover - only needed for vector parameter adaptation.
        return
    register_vector(conn)


def postgres_search_sql(*, include_vector: bool) -> str:
    vector_select = ", 1 - (embedding <=> %s) AS vector_score" if include_vector else ", 0.0::double precision AS vector_score"
    vector_filter = "OR embedding IS NOT NULL" if include_vector else ""
    vector_evidence = (
        "CASE WHEN vector_score >= 0.78 THEN 'vector:cosine:' || round(vector_score::numeric, 4)::text END,"
        if include_vector
        else ""
    )
    return f"""
        WITH scored AS (
            SELECT source_id,
                   retailer,
                   brand_raw,
                   name_raw,
                   price_cents,
                   CASE
                       WHEN name_norm = %s OR source_sku = %s THEN 1.0
                       WHEN %s <> '' AND (
                           position(%s in name_norm) > 0
                           OR position(%s in search_text) > 0
                           OR position(%s in brand_norm) > 0
                       ) THEN 0.99
                       ELSE 0.0
                   END AS exact_score,
                   CASE
                       WHEN %s <> '' AND search_vector @@ plainto_tsquery('simple', %s)
                       THEN LEAST(1.0, ts_rank_cd(search_vector, plainto_tsquery('simple', %s)) * 1.4)
                       ELSE 0.0
                   END AS fts_score,
                   GREATEST(similarity(name_norm, %s), similarity(search_text, %s)) AS trigram_score
                   {vector_select}
            FROM cartsy_products
            WHERE name_norm % %s
               OR search_text % %s
               OR search_vector @@ plainto_tsquery('simple', %s)
               OR name_norm = %s
               OR source_sku = %s
               {vector_filter}
        )
        SELECT source_id,
               retailer,
               brand_raw,
               name_raw,
               price_cents,
               GREATEST(exact_score, fts_score, trigram_score, vector_score) AS score,
               ARRAY_REMOVE(ARRAY[
                   CASE WHEN exact_score >= 0.99 THEN 'exact:name_or_sku' END,
                   CASE WHEN fts_score > 0 THEN 'lexical:fts:' || round(fts_score::numeric, 4)::text END,
                   CASE WHEN trigram_score >= 0.30 THEN 'trigram:title:' || round(trigram_score::numeric, 4)::text END,
                   {vector_evidence}
                   'backend:postgres'
               ], NULL) AS evidence
        FROM scored
        ORDER BY score DESC, name_raw ASC
        LIMIT %s
    """


def get_group(run_dir: str | Path, dedupe_id: str) -> dict[str, object]:
    group = read_group(run_dir, dedupe_id)
    offers = [row for row in read_assignments(run_dir) if row.get("dedupe_id") == dedupe_id]
    if group is None and not offers:
        raise LookupError(f"No group found for dedupe_id={dedupe_id}")
    if group is None:
        group = {
            "dedupe_id": dedupe_id,
            "source_ids": [offer["source_id"] for offer in offers],
            "canonical_name": offers[0].get("canonical_name", ""),
            "canonical_brand": offers[0].get("canonical_brand", ""),
            "cluster_confidence": offers[0].get("cluster_confidence", ""),
            "num_offers": len(offers),
            "retailers": sorted({offer.get("retailer", "") for offer in offers if offer.get("retailer")}),
        }
    group = dict(group)
    group["offers"] = [
        {
            "source_id": offer.get("source_id", ""),
            "retailer": offer.get("retailer", ""),
            "name": offer.get("name_raw", ""),
            "brand": offer.get("brand_raw", ""),
            "price_cents": offer.get("price_cents", ""),
            "sku": offer.get("sku", ""),
            "dimension": offer.get("dimension", ""),
        }
        for offer in offers
    ]
    return group


def explain_pair(run_dir: str | Path, source_id_a: str, source_id_b: str) -> dict[str, object]:
    left, right = sorted([source_id_a, source_id_b])
    pair = find_candidate_pair(run_dir, left, right)
    assignments = {row["source_id"]: row for row in read_assignments(run_dir)}
    if pair is None:
        return {
            "found": False,
            "product_a": assignments.get(source_id_a, {"source_id": source_id_a}),
            "product_b": assignments.get(source_id_b, {"source_id": source_id_b}),
            "message": "Pair was not generated as a candidate in this run.",
        }
    return {
        "found": True,
        "product_a": assignments.get(source_id_a, {"source_id": source_id_a}),
        "product_b": assignments.get(source_id_b, {"source_id": source_id_b}),
        "pair": pair,
    }


def read_assignments(run_dir: str | Path) -> list[dict[str, str]]:
    path = Path(run_dir) / "product_assignments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_group(run_dir: str | Path, dedupe_id: str) -> dict[str, object] | None:
    path = Path(run_dir) / "dedupe_groups.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            group = json.loads(line)
            if group.get("dedupe_id") == dedupe_id:
                return group
    return None


def find_candidate_pair(run_dir: str | Path, source_id_a: str, source_id_b: str) -> dict[str, object] | None:
    run_path = Path(run_dir)
    parquet_path = run_path / "candidate_pairs.parquet"
    csv_path = run_path / "candidate_pairs.csv"
    if parquet_path.exists():
        try:
            import polars as pl

            df = pl.read_parquet(parquet_path)
            found = df.filter(
                (
                    (pl.col("product_a_id").cast(pl.Utf8) == source_id_a)
                    & (pl.col("product_b_id").cast(pl.Utf8) == source_id_b)
                )
                | (
                    (pl.col("product_a_id").cast(pl.Utf8) == source_id_b)
                    & (pl.col("product_b_id").cast(pl.Utf8) == source_id_a)
                )
            )
            if found.height:
                return found.row(0, named=True)
        except ImportError:
            pass
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pair_ids = {row.get("product_a_id", ""), row.get("product_b_id", "")}
                if pair_ids == {source_id_a, source_id_b}:
                    return row
    return None


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        print("No results.")
        return
    widths = {
        column: min(
            48,
            max(len(column), *(len(_stringify(row.get(column, ""))) for row in rows)),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(
                truncate(_stringify(row.get(column, "")), widths[column]).ljust(widths[column])
                for column in columns
            )
        )


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def _stringify(value: object) -> str:
    return "" if value is None else str(value)
