from __future__ import annotations

from collections import Counter

from .schemas import CandidatePair, NormalizedProduct


def build_summary_report(
    *,
    products: list[NormalizedProduct],
    candidate_pairs: list[CandidatePair],
    clusters: dict[str, dict[str, object]],
    blocking_stats: dict[str, int],
    elapsed_seconds: float,
) -> dict[str, object]:
    decisions = Counter(pair.decision for pair in candidate_pairs)
    grouped_records = sum(int(cluster["num_offers"]) for cluster in clusters.values() if int(cluster["num_offers"]) > 1)
    duplicate_records_grouped = sum(int(cluster["num_offers"]) - 1 for cluster in clusters.values() if int(cluster["num_offers"]) > 1)
    quality_flags = Counter(flag for product in products for flag in product.quality_flags)
    confidence_values = [pair.score for pair in candidate_pairs]

    return {
        "input_records": len(products),
        "normalized_records": len(products),
        "candidate_pairs_generated": sum(decisions.values()),
        "auto_merged_pairs": decisions.get("auto_merge", 0),
        "review_pairs": decisions.get("review", 0),
        "final_unique_products": len(clusters),
        "grouped_records": grouped_records,
        "duplicate_records_grouped": duplicate_records_grouped,
        "reduction_ratio": round(duplicate_records_grouped / len(products), 4) if products else 0.0,
        "confidence_distribution": confidence_distribution(confidence_values),
        "top_quality_flags": dict(quality_flags.most_common(20)),
        "blocking": blocking_stats,
        "largest_groups": largest_groups(clusters),
        "lowest_confidence_accepted_merges": lowest_confidence_groups(clusters),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def confidence_distribution(values: list[float]) -> dict[str, int]:
    buckets = {
        "0.95-1.00": 0,
        "0.90-0.95": 0,
        "0.86-0.90": 0,
        "0.70-0.86": 0,
        "<0.70": 0,
    }
    for value in values:
        if value >= 0.95:
            buckets["0.95-1.00"] += 1
        elif value >= 0.90:
            buckets["0.90-0.95"] += 1
        elif value >= 0.86:
            buckets["0.86-0.90"] += 1
        elif value >= 0.70:
            buckets["0.70-0.86"] += 1
        else:
            buckets["<0.70"] += 1
    return buckets


def largest_groups(clusters: dict[str, dict[str, object]], limit: int = 10) -> list[dict[str, object]]:
    ordered = sorted(clusters.values(), key=lambda cluster: int(cluster["num_offers"]), reverse=True)
    return [
        {
            "dedupe_id": cluster["dedupe_id"],
            "canonical_name": cluster["canonical_name"],
            "num_offers": cluster["num_offers"],
            "retailers": cluster["retailers"],
            "cluster_confidence": round(float(cluster["cluster_confidence"]), 4),
            "source_ids": cluster["source_ids"][:10],
        }
        for cluster in ordered[:limit]
    ]


def lowest_confidence_groups(clusters: dict[str, dict[str, object]], limit: int = 10) -> list[dict[str, object]]:
    grouped = [cluster for cluster in clusters.values() if int(cluster["num_offers"]) > 1]
    ordered = sorted(grouped, key=lambda cluster: float(cluster["cluster_confidence"]))
    return [
        {
            "dedupe_id": cluster["dedupe_id"],
            "canonical_name": cluster["canonical_name"],
            "num_offers": cluster["num_offers"],
            "cluster_confidence": round(float(cluster["cluster_confidence"]), 4),
            "source_ids": cluster["source_ids"][:10],
            "merge_reasons": cluster["merge_reasons"],
        }
        for cluster in ordered[:limit]
    ]
