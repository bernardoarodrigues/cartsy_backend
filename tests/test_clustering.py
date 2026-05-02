from __future__ import annotations

from cartsy_dedupe.clustering import build_clusters
from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.schemas import CandidatePair


def product(source_id: str):
    return normalize_row(
        {
            "id": source_id,
            "prod_name": "Cetaphil Loção Hidratante 473ml",
            "brand": "Cetaphil",
            "category": "Beleza>Pele",
            "description": "",
            "specs": "{}",
            "img_links": "",
            "url": "",
            "created_at": "",
            "updated_at": "",
            "retailer": "amazon_br",
            "price": "6790",
            "sku": "",
            "dimension": "473ml",
        }
    )


def product_with_dimension(source_id: str, dimension: str):
    return normalize_row(
        {
            "id": source_id,
            "prod_name": f"Cetaphil Loção Hidratante {dimension}",
            "brand": "Cetaphil",
            "category": "Beleza>Pele",
            "description": "",
            "specs": "{}",
            "img_links": "",
            "url": "",
            "created_at": "",
            "updated_at": "",
            "retailer": "amazon_br",
            "price": "6790",
            "sku": "",
            "dimension": dimension,
        }
    )


def test_merge_edges_form_cluster() -> None:
    products = [product("1"), product("2"), product("3")]
    pairs = [
        CandidatePair("1", "2", 0.92, "merge", "brand_match", ("x",), {}),
        CandidatePair("2", "3", 0.80, "no_merge", "close", ("x",), {}),
    ]
    clusters, stats = build_clusters(products, pairs, {"1": 0, "2": 1, "3": 2})
    grouped = [cluster for cluster in clusters.values() if int(cluster["num_offers"]) == 2]
    assert len(grouped) == 1
    assert grouped[0]["cluster_confidence"] == 0.92
    assert stats["merge_edges_accepted"] == 1


def test_cluster_guard_blocks_incompatible_sizes_even_if_edge_says_merge() -> None:
    products = [product_with_dimension("1", "200ml"), product_with_dimension("2", "473ml")]
    pairs = [
        CandidatePair("1", "2", 0.92, "merge", "forced", ("x",), {}),
    ]
    clusters, stats = build_clusters(products, pairs, {"1": 0, "2": 1})
    assert all(int(cluster["num_offers"]) == 1 for cluster in clusters.values())
    assert stats["merge_edges_blocked_by_cluster_guard"] == 1

