from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from .blocking import generate_candidate_pairs
from .clustering import build_clusters
from .config import PipelineConfig
from .ingest import load_normalized_products
from .reporting import build_summary_report
from .schemas import CandidatePair, NormalizedProduct
from .scoring import score_pair
from .storage import prepare_output_dir, write_outputs


def run_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    limit: int | None = None,
) -> dict[str, object]:
    started = perf_counter()
    output_path = prepare_output_dir(output_dir)

    print(f"loading and normalizing {input_path}")
    products = load_normalized_products(input_path, limit=limit)
    print(f"normalized {len(products):,} products")
    id_to_index = {product.source_id: index for index, product in enumerate(products)}

    print("generating candidate pairs")
    pair_blocks, blocking_stats = generate_candidate_pairs(
        products,
        max_block_size=config.max_block_size,
        max_candidate_pairs=config.max_candidate_pairs,
    )
    print(f"generated {len(pair_blocks):,} candidate pairs")

    print("scoring candidate pairs")
    candidate_pairs: list[CandidatePair] = []
    for pair_number, ((left_index, right_index), block_keys) in enumerate(pair_blocks.items(), start=1):
        left = products[left_index]
        right = products[right_index]
        result = score_pair(
            left,
            right,
            auto_threshold=config.auto_threshold,
            review_threshold=config.review_threshold,
        )
        if result.decision == "reject" and result.score < config.review_threshold:
            continue
        candidate_pairs.append(
            CandidatePair(
                product_a_id=left.source_id,
                product_b_id=right.source_id,
                score=result.score,
                decision=result.decision,
                explanation=result.explanation,
                blocking_keys=tuple(sorted(block_keys)),
                feature_scores=result.feature_scores,
            )
        )
        if pair_number % 100_000 == 0:
            print(f"scored {pair_number:,} candidate pairs; kept {len(candidate_pairs):,}")

    clusters = build_clusters(products, candidate_pairs, id_to_index)
    source_to_cluster = invert_clusters(clusters)
    report = build_summary_report(
        products=products,
        candidate_pairs=candidate_pairs,
        clusters=clusters,
        blocking_stats=blocking_stats,
        elapsed_seconds=perf_counter() - started,
    )
    write_outputs(
        output_path=output_path,
        products=products,
        candidate_pairs=candidate_pairs,
        clusters=clusters,
        source_to_cluster=source_to_cluster,
        report=report,
        review_limit=config.review_limit,
        sample_pair_limit=config.sample_pair_limit,
    )
    return report


def invert_clusters(clusters: dict[str, dict[str, object]]) -> dict[str, str]:
    source_to_cluster: dict[str, str] = {}
    for dedupe_id, cluster in clusters.items():
        for source_id in cluster["source_ids"]:
            source_to_cluster[str(source_id)] = dedupe_id
    return source_to_cluster


def decision_counts(candidate_pairs: list[CandidatePair]) -> Counter[str]:
    return Counter(pair.decision for pair in candidate_pairs)


def products_by_source_id(products: list[NormalizedProduct]) -> dict[str, NormalizedProduct]:
    return {product.source_id: product for product in products}
