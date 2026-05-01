from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.text import informative_tokens, normalize_text

T = TypeVar("T")


class ExtractedAttributes(BaseModel):
    brand: str | None = None
    product_line: str | None = None
    product_type: str | None = None
    category: str | None = None
    color: str | None = None
    size: str | None = None
    scent: str | None = None
    flavor: str | None = None
    material: str | None = None
    pack_count: str | None = None
    variant_name: str | None = None
    model_number: str | None = None
    sku_like_identifiers: list[str] = Field(default_factory=list)


def exact_keys(product: NormalizedProduct) -> dict[str, str]:
    keys: dict[str, str] = {}
    for key in ("ean", "gtin", "upc", "asin"):
        value = product.identifiers.get(key)
        if value:
            keys[key] = value
    if product.retailer and product.identifiers.get("sku"):
        keys[f"retailer_sku:{product.retailer}"] = product.identifiers["sku"]
    url_key = canonicalize_url(product.url)
    if url_key:
        keys["canonical_url"] = url_key
    return keys


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/").lower()
    if not host or not path:
        return ""
    return normalize_text(f"{host} {path}").replace(" ", "/")[:240]


def product_search_text(product: NormalizedProduct) -> str:
    tokens = informative_tokens(product.name_norm, limit=8)
    return " ".join(
        part
        for part in [
            product.brand_norm,
            " ".join(tokens),
            product.category_leaf,
            product.dimension_raw,
        ]
        if part
    )


def embedding_text(**parts: str | None) -> str:
    return "\n".join(f"{key}: {value}" for key, value in parts.items() if value)


def ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env or the environment before running the postgres_openai pipeline.")


def batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for index in range(0, len(items), max(1, size)):
        yield items[index : index + size]


def extracted_attribute_score(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[float, str, list[str]]:
    if not left or not right:
        return 0.45, "unknown", []
    positive = 0
    comparable = 0
    conflicts: list[str] = []
    reasons: list[str] = []
    for key in (
        "brand",
        "product_line",
        "product_type",
        "category",
        "variant_name",
        "color",
        "size",
        "scent",
        "flavor",
        "material",
        "pack_count",
        "model_number",
    ):
        left_value = normalize_text(left.get(key))
        right_value = normalize_text(right.get(key))
        if not left_value or not right_value:
            continue
        comparable += 1
        if left_value == right_value:
            positive += 1
            reasons.append(f"llm_{key}_match:{left_value}")
        elif key in {"variant_name", "color", "size", "scent", "flavor", "material", "pack_count", "model_number"}:
            conflicts.append(f"llm_{key}_conflict")
    if comparable == 0:
        return 0.45, "unknown", reasons
    score = positive / comparable
    relation = "same_parent_different_variant" if conflicts and same_parent_attributes(left, right) else "unknown"
    reasons.extend(conflicts)
    return score, relation, reasons


def same_parent_attributes(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("brand", "product_line", "product_type"):
        left_value = normalize_text(left.get(key))
        right_value = normalize_text(right.get(key))
        if left_value and right_value and left_value != right_value:
            return False
    return bool(normalize_text(left.get("product_line")) and normalize_text(right.get("product_line")))


def invert_clusters(clusters: dict[str, dict[str, object]]) -> dict[str, str]:
    source_to_cluster: dict[str, str] = {}
    for dedupe_id, cluster in clusters.items():
        for source_id in cluster["source_ids"]:
            source_to_cluster[str(source_id)] = dedupe_id
    return source_to_cluster
