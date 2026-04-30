from __future__ import annotations

import hashlib
from collections import defaultdict

from cartsy_dedupe.config import GENERIC_BRANDS, GLOBAL_IDENTIFIER_KEYS
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct
from cartsy_dedupe.text import normalize_text

from .attributes import sizes_equivalent


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
        self.members = {index: [index] for index in range(size)}

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
        self.members[root_left].extend(self.members.pop(root_right))
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True


def build_clusters(
    products: list[NormalizedProduct],
    candidate_pairs: list[CandidatePair],
    id_to_index: dict[str, int],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    uf = UnionFind(len(products))
    accepted_edges: list[CandidatePair] = []
    blocked_edges = 0
    for pair in candidate_pairs:
        if pair.decision != "merge":
            continue
        left = id_to_index[pair.product_a_id]
        right = id_to_index[pair.product_b_id]
        left_cluster = indexes_for_root(uf, left)
        right_cluster = indexes_for_root(uf, right)
        if left_cluster != right_cluster and has_cluster_contradiction(products, left_cluster + right_cluster):
            blocked_edges += 1
            continue
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
    return clusters, {
        "merge_edges_accepted": len(accepted_edges),
        "merge_edges_blocked_by_cluster_guard": blocked_edges,
    }


def indexes_for_root(uf: UnionFind, index: int) -> list[int]:
    root = uf.find(index)
    return uf.members[root]


def has_cluster_contradiction(products: list[NormalizedProduct], indexes: list[int]) -> bool:
    members = [products[index] for index in indexes]
    brands = {product.brand_norm for product in members if product.brand_norm and product.brand_norm not in GENERIC_BRANDS}
    if len(brands) > 1:
        return True

    for key in GLOBAL_IDENTIFIER_KEYS:
        values = {product.identifiers[key] for product in members if product.identifiers.get(key)}
        if len(values) > 1:
            return True

    if has_size_contradiction(members):
        return True

    for attr in ("pack_count",):
        values = {getattr(product, attr) for product in members if getattr(product, attr)}
        if len(values) > 1:
            return True
    if has_extracted_attribute_contradiction(members):
        return True
    return False


def has_size_contradiction(members: list[NormalizedProduct]) -> bool:
    sizes = [
        (product.size_value, product.size_unit)
        for product in members
        if product.size_value is not None and product.size_unit and not product.size_ambiguous
    ]
    if len(sizes) < 2:
        return False
    first_value, first_unit = sizes[0]
    for value, unit in sizes[1:]:
        if first_value is None or value is None or unit != first_unit:
            return True
        if not sizes_equivalent(first_value, first_unit, value, unit):
            return True
    return False


def has_extracted_attribute_contradiction(members: list[NormalizedProduct]) -> bool:
    for key in ("product_line", "product_type", "variant_name", "color", "size", "scent", "flavor", "material", "pack_count", "model_number"):
        values = {
            normalize_text(product.extracted_attributes.get(key))
            for product in members
            if product.extracted_attributes.get(key)
        }
        if len(values) > 1:
            return True
    return False


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
