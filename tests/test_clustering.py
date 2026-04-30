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


def test_merge_edges_form_cluster() -> None:
    products = [product("1"), product("2"), product("3")]
    pairs = [
        CandidatePair("1", "2", 0.92, "merge", "brand_match", ("x",), {}),
        CandidatePair("2", "3", 0.80, "no_merge", "close", ("x",), {}),
    ]
    clusters = build_clusters(products, pairs, {"1": 0, "2": 1, "3": 2})
    grouped = [cluster for cluster in clusters.values() if int(cluster["num_offers"]) == 2]
    assert len(grouped) == 1
    assert grouped[0]["cluster_confidence"] == 0.92
