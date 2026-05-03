from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar
from urllib.parse import urlparse

from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.text import informative_tokens, normalize_text

T = TypeVar("T")


def exact_keys(product: NormalizedProduct) -> dict[str, str]:
    """Build exact-match keys from identifiers and trusted URLs."""
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
    """Normalize product URLs into stable match keys when trustworthy."""
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/").lower()
    if not host or not path or not trustworthy_product_url(host, path):
        return ""
    return normalize_text(f"{host} {path}").replace(" ", "/")[:240]


def trustworthy_product_url(host: str, path: str) -> bool:
    """Keep exact URL keys to product pages, not click/redirect/tracking links."""
    combined = f"{host}/{path}".lower()
    redirect_tokens = (
        "click",
        "count",
        "redirect",
        "redir",
        "goto",
        "tracking",
        "track",
        "affiliate",
        "afiliado",
        "adservice",
        "ads",
    )
    if any(token in combined for token in redirect_tokens):
        return False
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return False
    text = normalize_text(" ".join(segments))
    tokens = [token for token in text.split() if len(token) >= 3]
    has_product_id = any(any(char.isdigit() for char in token) for token in tokens)
    has_descriptive_slug = len(tokens) >= 2 and sum(len(token) for token in tokens) >= 12
    return has_product_id or has_descriptive_slug


def product_search_text(product: NormalizedProduct) -> str:
    """Build weighted text used for FTS and artifact search."""
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
    """Build the text sent to the embedding backend for one product."""
    return "\n".join(f"{key}: {value}" for key, value in parts.items() if value)


def batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield fixed-size batches from a sequence."""
    for index in range(0, len(items), max(1, size)):
        yield items[index : index + size]


def invert_clusters(clusters: dict[str, dict[str, object]]) -> dict[str, str]:
    """Map source ids back to their dedupe cluster ids."""
    source_to_cluster: dict[str, str] = {}
    for dedupe_id, cluster in clusters.items():
        for source_id in cluster["source_ids"]:
            source_to_cluster[str(source_id)] = dedupe_id
    return source_to_cluster
