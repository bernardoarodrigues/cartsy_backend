from __future__ import annotations

import hashlib
from collections import defaultdict

from .schemas import CandidatePair, NormalizedProduct


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True


def build_clusters(
    products: list[NormalizedProduct],
    candidate_pairs: list[CandidatePair],
    id_to_index: dict[str, int],
) -> dict[str, dict[str, object]]:
    uf = UnionFind(len(products))
    accepted_edges: list[CandidatePair] = []
    for pair in candidate_pairs:
        if pair.decision != "merge":
            continue
        left = id_to_index[pair.product_a_id]
        right = id_to_index[pair.product_b_id]
        uf.union(left, right)
        accepted_edges.append(pair)

    root_to_indexes: dict[int, list[int]] = defaultdict(list)
    for index in range(len(products)):
        root_to_indexes[uf.find(index)].append(index)

    edge_scores_by_root: dict[int, list[float]] = defaultdict(list)
    reasons_by_root: dict[int, list[str]] = defaultdict(list)
    for edge in accepted_edges:
        root = uf.find(id_to_index[edge.product_a_id])
        edge_scores_by_root[root].append(edge.score)
        if edge.explanation:
            reasons_by_root[root].append(edge.explanation)

    clusters: dict[str, dict[str, object]] = {}
    for root, indexes in root_to_indexes.items():
        source_ids = sorted(products[index].source_id for index in indexes)
        dedupe_id = stable_dedupe_id(source_ids)
        members = [products[index] for index in indexes]
        scores = edge_scores_by_root.get(root, [])
        clusters[dedupe_id] = {
            "dedupe_id": dedupe_id,
            "source_ids": source_ids,
            "indexes": indexes,
            "canonical_name": choose_canonical_name(members),
            "canonical_brand": choose_mode([product.brand_raw or product.brand_norm for product in members]),
            "canonical_category": choose_mode([product.category_raw or product.category_norm for product in members]),
            "cluster_confidence": min(scores) if scores else 1.0,
            "num_offers": len(indexes),
            "retailers": sorted({product.retailer for product in members if product.retailer}),
            "price_min_cents": min((product.price_cents for product in members if product.price_cents is not None), default=None),
            "price_max_cents": max((product.price_cents for product in members if product.price_cents is not None), default=None),
            "merge_reasons": reasons_by_root.get(root, [])[:5],
        }
    return clusters


def stable_dedupe_id(source_ids: list[str]) -> str:
    digest = hashlib.sha1("|".join(source_ids).encode("utf-8")).hexdigest()[:12]
    return f"prod_{digest}"


def choose_canonical_name(products: list[NormalizedProduct]) -> str:
    def quality(product: NormalizedProduct) -> tuple[int, int]:
        contains_brand = int(bool(product.brand_norm and product.brand_norm in product.name_norm))
        return contains_brand, len(product.name_raw)

    return max(products, key=quality).name_raw


def choose_mode(values: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    first_seen: dict[str, int] = {}
    for index, value in enumerate(values):
        if not value:
            continue
        normalized = value.strip()
        counts[normalized] += 1
        first_seen.setdefault(normalized, index)
    if not counts:
        return ""
    return max(counts, key=lambda value: (counts[value], -first_seen[value]))
