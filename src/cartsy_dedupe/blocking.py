from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .config import GENERIC_BRANDS
from .schemas import NormalizedProduct
from .text import informative_tokens


def product_blocking_keys(product: NormalizedProduct) -> tuple[str, ...]:
    keys: set[str] = set()

    for key in ("ean", "gtin", "upc"):
        value = product.identifiers.get(key)
        if value:
            keys.add(f"id:{key}:{value}")

    asin = product.identifiers.get("asin")
    if asin:
        keys.add(f"id:asin:{asin}")

    sku = product.identifiers.get("sku")
    if sku:
        keys.add(f"retailer_sku:{product.retailer}:{sku}")

    brand = product.brand_norm
    if brand and brand not in GENERIC_BRANDS:
        if product.name_norm:
            keys.add(f"brand_name:{brand}:{product.name_norm[:120]}")

        for model in product.model_tokens[:3]:
            keys.add(f"brand_model:{brand}:{model}")

        title_tokens = informative_tokens(product.name_norm, limit=4)
        if len(title_tokens) >= 3:
            keys.add(f"brand_title:{brand}:{' '.join(title_tokens[:3])}")

        if product.category_leaf and len(title_tokens) >= 2:
            keys.add(f"brand_cat_title:{brand}:{product.category_leaf}:{' '.join(title_tokens[:2])}")

        if product.size_value is not None and product.size_unit and product.category_leaf:
            rounded_size = round(product.size_value, 0 if product.size_value >= 10 else 1)
            keys.add(f"brand_size_cat:{brand}:{rounded_size:g}{product.size_unit}:{product.category_leaf}")

    return tuple(sorted(keys))


def generate_candidate_pairs(
    products: list[NormalizedProduct],
    *,
    max_block_size: int,
    max_candidate_pairs: int | None,
) -> tuple[dict[tuple[int, int], set[str]], dict[str, int]]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, product in enumerate(products):
        for key in product_blocking_keys(product):
            blocks[key].append(index)

    pairs: dict[tuple[int, int], set[str]] = defaultdict(set)
    skipped_blocks = 0
    oversized_rows = 0
    for key, indexes in blocks.items():
        if len(indexes) < 2:
            continue
        if len(indexes) > max_block_size:
            skipped_blocks += 1
            oversized_rows += len(indexes)
            continue
        for left, right in combinations(indexes, 2):
            if left == right:
                continue
            if left > right:
                left, right = right, left
            pairs[(left, right)].add(key)
            if max_candidate_pairs is not None and len(pairs) >= max_candidate_pairs:
                return pairs, {
                    "blocking_keys": len(blocks),
                    "skipped_blocks": skipped_blocks,
                    "oversized_block_rows": oversized_rows,
                    "candidate_cap_reached": 1,
                }

    return pairs, {
        "blocking_keys": len(blocks),
        "skipped_blocks": skipped_blocks,
        "oversized_block_rows": oversized_rows,
        "candidate_cap_reached": 0,
    }
