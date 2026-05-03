from __future__ import annotations

import csv
import json
from pathlib import Path

from .schemas import CandidatePair, NormalizedProduct

try:
    import polars as pl
except ImportError:  # pragma: no cover - depends on local environment.
    pl = None


def prepare_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_outputs(
    *,
    output_path: Path,
    products: list[NormalizedProduct],
    candidate_pairs: list[CandidatePair],
    clusters: dict[str, dict[str, object]],
    source_to_cluster: dict[str, str],
    report: dict[str, object],
    near_miss_limit: int,
    sample_pair_limit: int,
) -> None:
    write_table(output_path / "normalized_products.parquet", [product.to_record() for product in products])
    write_table(
        output_path / "candidate_pairs.parquet",
        [pair.to_record() for pair in candidate_pairs[:sample_pair_limit]],
    )
    write_product_assignments(output_path / "product_assignments.csv", products, clusters, source_to_cluster)
    write_groups(output_path / "dedupe_groups.jsonl", clusters)
    write_near_misses(output_path / "near_miss_pairs.csv", products, candidate_pairs, near_miss_limit)
    (output_path / "summary_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if pl is not None:
        pl.DataFrame(rows).write_parquet(path)
        return

    fallback = path.with_suffix(".csv")
    write_csv(fallback, rows)
    marker = {
        "message": "polars is not installed; wrote CSV fallback instead of parquet",
        "fallback": fallback.name,
    }
    path.with_suffix(path.suffix + ".fallback.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_groups(path: Path, clusters: dict[str, dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for cluster in sorted(clusters.values(), key=lambda item: str(item["dedupe_id"])):
            payload = {key: value for key, value in cluster.items() if key != "indexes"}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_product_assignments(
    path: Path,
    products: list[NormalizedProduct],
    clusters: dict[str, dict[str, object]],
    source_to_cluster: dict[str, str],
) -> None:
    rows: list[dict[str, object]] = []
    for product in products:
        dedupe_id = source_to_cluster[product.source_id]
        cluster = clusters[dedupe_id]
        rows.append(
            {
                "source_id": product.source_id,
                "dedupe_id": dedupe_id,
                "retailer": product.retailer,
                "name_raw": product.name_raw,
                "brand_raw": product.brand_raw,
                "price_cents": product.price_cents if product.price_cents is not None else "",
                "sku": product.source_sku,
                "dimension": product.dimension_raw,
                "canonical_name": cluster["canonical_name"],
                "canonical_brand": cluster["canonical_brand"],
                "cluster_confidence": round(float(cluster["cluster_confidence"]), 4),
                "decision": "grouped" if int(cluster["num_offers"]) > 1 else "singleton",
                "explanation": " | ".join(str(reason) for reason in cluster["merge_reasons"][:2]),
            }
        )
    write_csv(path, rows)


def write_near_misses(
    path: Path,
    products: list[NormalizedProduct],
    candidate_pairs: list[CandidatePair],
    limit: int,
) -> None:
    product_by_id = {product.source_id: product for product in products}
    near_miss_pairs = sorted(
        [pair for pair in candidate_pairs if pair.decision == "no_merge"],
        key=lambda pair: pair.score,
        reverse=True,
    )[:limit]
    rows: list[dict[str, object]] = []
    for pair in near_miss_pairs:
        left = product_by_id[pair.product_a_id]
        right = product_by_id[pair.product_b_id]
        rows.append(
            {
                "product_a_id": pair.product_a_id,
                "product_b_id": pair.product_b_id,
                "score": round(pair.score, 4),
                "ml_score": round(pair.ml_score, 4),
                "evidence_score": round(pair.evidence_score, 4),
                "decision_threshold": round(pair.decision_threshold, 4),
                "decision_reason": pair.decision_reason,
                "decision": pair.decision,
                "name_a": left.name_raw,
                "name_b": right.name_raw,
                "brand_a": left.brand_raw,
                "brand_b": right.brand_raw,
                "retailer_a": left.retailer,
                "retailer_b": right.retailer,
                "price_a": left.price_cents if left.price_cents is not None else "",
                "price_b": right.price_cents if right.price_cents is not None else "",
                "dimension_a": left.dimension_raw,
                "dimension_b": right.dimension_raw,
                "explanation": pair.explanation,
            }
        )
    write_csv(path, rows)
