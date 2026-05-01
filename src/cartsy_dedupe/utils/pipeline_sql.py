from __future__ import annotations


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
        WITH brand_sizes AS (
            SELECT brand_norm, COUNT(*)::int AS brand_size
            FROM cartsy_products
            WHERE brand_norm <> ''
            GROUP BY brand_norm
        )
        SELECT p.source_index, q.source_index,
               'trigram:title:' || round(q.similarity::numeric, 4)::text AS evidence
        FROM cartsy_products p
        JOIN brand_sizes b
          ON b.brand_norm = p.brand_norm
        JOIN LATERAL (
            SELECT candidate.source_index,
                   1 - (candidate.name_norm <-> p.name_norm) AS similarity
            FROM cartsy_products candidate
            WHERE candidate.source_index > p.source_index
              AND candidate.brand_norm = p.brand_norm
              AND candidate.brand_norm <> ''
              AND p.name_norm <> ''
              AND abs(char_length(candidate.name_norm) - char_length(p.name_norm)) <= 24
              AND (candidate.name_norm <-> p.name_norm) <= (1 - %s)
            ORDER BY candidate.name_norm <-> p.name_norm ASC
            LIMIT %s
        ) q ON true
        WHERE b.brand_size <= COALESCE(%s, b.brand_size)
          AND q.similarity >= 0.45
    """


def vector_candidate_sql() -> str:
    return """
        WITH raw AS (
            SELECT LEAST(p.source_index, q.source_index) AS left_index,
                   GREATEST(p.source_index, q.source_index) AS right_index,
                   q.similarity
            FROM cartsy_products p
            JOIN LATERAL (
                SELECT candidate.source_index,
                       1 - (candidate.embedding <=> p.embedding) AS similarity
                FROM cartsy_products candidate
                WHERE p.embedding IS NOT NULL
                  AND candidate.embedding IS NOT NULL
                  AND candidate.source_index = ANY(%s)
                  AND candidate.source_index <> p.source_index
                ORDER BY candidate.embedding <=> p.embedding
                LIMIT %s
            ) q ON true
            WHERE p.source_index = ANY(%s)
        )
        SELECT left_index,
               right_index,
               'vector:cosine:' || round(MAX(similarity)::numeric, 4)::text AS evidence
        FROM raw
        WHERE similarity >= 0.78
        GROUP BY left_index, right_index
    """


def evidence_value(key: str, *, default: float) -> float:
    try:
        return float(key.rsplit(":", 1)[1])
    except ValueError:
        return default


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
