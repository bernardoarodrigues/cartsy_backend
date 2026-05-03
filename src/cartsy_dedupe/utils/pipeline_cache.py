from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct

CACHE_SCHEMA_VERSION = 3


def stage_cache_enabled() -> bool:
    """Return whether stage-level pipeline caching is enabled."""
    raw = os.getenv("CARTSY_STAGE_CACHE_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def embedding_cache_enabled() -> bool:
    """Return whether product embedding caching is enabled."""
    raw = os.getenv("CARTSY_EMBEDDING_CACHE_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def pipeline_cache_root() -> Path:
    """Return the root directory for local pipeline caches."""
    base = os.getenv("CARTSY_PIPELINE_CACHE_DIR")
    if base:
        return Path(base)
    return Path(".cache") / "cartsy-dedupe"


def normalization_cache_dir() -> Path:
    """Return the cache directory for normalized product rows."""
    return pipeline_cache_root() / "normalization"


def stage_cache_dir(stage_name: str) -> Path:
    """Return the cache directory for stage payloads."""
    path = pipeline_cache_root() / stage_name
    return path


def embedding_cache_dir() -> Path:
    """Return embedding cache dir."""
    return stage_cache_dir("embeddings")


def file_sha256(path: Path) -> str:
    """Compute a SHA-256 fingerprint for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint(*relative_paths: str) -> dict[str, str]:
    """Fingerprint source files that affect a cached stage."""
    package_root = Path(__file__).resolve().parents[1]
    return {
        relative_path: file_sha256(package_root / relative_path)
        for relative_path in relative_paths
    }


def normalize_module_hash() -> str:
    """Fingerprint the normalization code path."""
    return file_sha256(Path(__file__).resolve().parents[1] / "normalize.py")


def stage_env_fingerprint(names: list[str]) -> dict[str, str | None]:
    """Capture environment variables that affect a cache key."""
    return {name: os.getenv(name) for name in names}


def cache_key(payload: dict[str, Any]) -> str:
    """Hash cache metadata into a stable cache key."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalization_cache_key(*, input_path: Path, limit: int | None, normalize_hash: str) -> str:
    """Build the cache key for normalized product rows."""
    stat = input_path.stat()
    key_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": "normalization",
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "limit": limit,
        "normalize_hash": normalize_hash,
    }
    return cache_key(key_payload)


def retrieval_cache_key(
    *,
    normalization_key: str,
    config: PipelineConfig,
    env: dict[str, str | None],
    code: dict[str, str],
) -> str:
    """Extract retrieval cache key."""
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": "retrieve_candidates",
            "normalization_key": normalization_key,
            "config": asdict(config),
            "env": env,
            "code": code,
        }
    )


def retrieval_layer_cache_key(
    *,
    normalization_key: str,
    layer: str,
    layer_params: dict[str, Any],
    env: dict[str, str | None],
    code: dict[str, str],
) -> str:
    """Extract retrieval layer cache key."""
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": f"retrieve_candidates:{layer}",
            "normalization_key": normalization_key,
            "layer": layer,
            "layer_params": layer_params,
            "env": env,
            "code": code,
        }
    )


def embedding_cache_key(
    *,
    normalization_key: str,
    embedding_provider: str = "openai",
    embedding_model: str,
    embedding_dimensions: int,
    code: dict[str, str],
) -> str:
    """Return embedding cache key."""
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": "product_embeddings",
            "normalization_key": normalization_key,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "code": code,
        }
    )


def shared_embedding_cache_key(
    *,
    embedding_provider: str = "openai",
    embedding_model: str,
    embedding_dimensions: int,
    code: dict[str, str],
) -> str:
    """Return the shared product embedding cache key used by training and dedupe."""
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": "product_embeddings_shared",
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "code": code,
        }
    )


def scoring_cache_key(
    *,
    retrieval_key: str,
    config: PipelineConfig,
    code: dict[str, str],
) -> str:
    """Build the cache key for scored candidate pairs."""
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": "score_candidates",
            "retrieval_key": retrieval_key,
            "config": asdict(config),
            "code": code,
        }
    )


def clustering_cache_key(
    *,
    scoring_key: str,
    code: dict[str, str],
) -> str:
    """Build the cache key for clustered output groups."""
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": "cluster",
            "scoring_key": scoring_key,
            "code": code,
        }
    )


def cache_path_for(stage_name: str, key: str) -> Path:
    """Resolve the cache file path for a key and extension."""
    return stage_cache_dir(stage_name) / f"{key}.json"


def embedding_text_hash(text: str) -> str:
    """Return embedding text hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_embedding_matrix_cache(
    *,
    normalization_key: str,
    expected_dimensions: int,
) -> tuple[Path, dict[str, int], np.ndarray] | None:
    """Find a reusable embedding matrix cache file."""
    for matrix_cache in iter_embedding_matrix_caches(
        expected_dimensions=expected_dimensions,
        normalization_key=normalization_key,
    ):
        return matrix_cache
    return None


def iter_embedding_matrix_caches(
    *,
    expected_dimensions: int,
    normalization_key: str | None = None,
) -> list[tuple[Path, dict[str, int], np.ndarray]]:
    """Iterate over embedding matrix cache metadata files."""
    cache_dir = embedding_cache_dir()
    pattern = (
        f"embeddings_{normalization_key}_*.source_id_to_index.json"
        if normalization_key
        else "embeddings_*.source_id_to_index.json"
    )
    candidates = sorted(
        cache_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    caches: list[tuple[Path, dict[str, int], np.ndarray]] = []
    for index_path in candidates:
        matrix_path = index_path.with_name(index_path.name.removesuffix(".source_id_to_index.json") + ".npy")
        if not matrix_path.exists():
            continue
        try:
            source_id_to_index = {
                str(source_id): int(index)
                for source_id, index in json.loads(index_path.read_text(encoding="utf-8")).items()
            }
            matrix = np.load(matrix_path, mmap_mode="r")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if matrix.ndim != 2 or int(matrix.shape[1]) != expected_dimensions:
            continue
        caches.append((matrix_path, source_id_to_index, matrix))
    return caches


def product_signature(products: list[NormalizedProduct]) -> str:
    """Fingerprint normalized products for downstream cache keys."""
    payload = [asdict(product) for product in products]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def pair_blocks_to_records(pair_blocks: dict[tuple[int, int], set[str]]) -> list[dict[str, object]]:
    """Build pair blocks to records."""
    return [
        {
            "left_index": left,
            "right_index": right,
            "blocking_keys": sorted(block_keys),
        }
        for (left, right), block_keys in sorted(pair_blocks.items())
    ]


def retrieval_rows_to_records(rows: list[tuple[int, int, str]]) -> list[dict[str, object]]:
    """Extract retrieval rows to records."""
    return [
        {
            "left_index": left,
            "right_index": right,
            "evidence": evidence,
        }
        for left, right, evidence in rows
    ]


def retrieval_rows_from_records(records: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """Extract retrieval rows from records."""
    rows: list[tuple[int, int, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            (
                int(record["left_index"]),
                int(record["right_index"]),
                str(record["evidence"]),
            )
        )
    return rows


def pair_blocks_from_records(records: list[dict[str, Any]]) -> dict[tuple[int, int], set[str]]:
    """Build pair blocks from records."""
    return {
        (int(record["left_index"]), int(record["right_index"])): set(record.get("blocking_keys") or [])
        for record in records
        if isinstance(record, dict)
    }


def candidate_pairs_to_records(candidate_pairs: list[CandidatePair]) -> list[dict[str, Any]]:
    """Serialize candidate pairs for stage-cache storage."""
    return [asdict(pair) for pair in candidate_pairs]


def candidate_pairs_from_records(records: list[dict[str, Any]]) -> list[CandidatePair]:
    """Deserialize cached candidate pairs."""
    pairs: list[CandidatePair] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        payload = dict(record)
        payload["blocking_keys"] = tuple(payload.get("blocking_keys") or [])
        payload["feature_scores"] = {
            str(key): float(value)
            for key, value in dict(payload.get("feature_scores") or {}).items()
        }
        pairs.append(CandidatePair(**payload))
    return pairs


def clusters_to_records(clusters: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Serialize clusters for stage-cache storage."""
    return json.loads(json.dumps(clusters, ensure_ascii=False, sort_keys=True))


def write_stage_cache(
    path: Path,
    *,
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """Write a stage-cache payload with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "metadata": metadata,
        "payload": payload,
    }
    path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")


def read_stage_cache(path: Path) -> dict[str, Any] | None:
    """Read and validate a stage-cache payload."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict):
        return None
    payload = blob.get("payload")
    if not isinstance(payload, dict):
        return None
    metadata = blob.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return None
    return blob


def stage_cache_status(path: Path, key: str, *, enabled: bool | None = None) -> dict[str, object]:
    """Build the status block recorded in summary_report.json."""
    return {
        "enabled": int(stage_cache_enabled() if enabled is None else enabled),
        "used": 0,
        "path": str(path),
        "key": key,
    }


def read_cache_payload(path: Path, *, enabled: bool | None = None) -> dict[str, Any] | None:
    """Read a cache payload from disk when present."""
    if not (stage_cache_enabled() if enabled is None else enabled):
        return None
    blob = read_stage_cache(path)
    if blob is None:
        return None
    return blob["payload"]


def write_cache_payload(
    path: Path,
    *,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    enabled: bool | None = None,
) -> None:
    """Write a JSON cache payload atomically enough for local reuse."""
    if not (stage_cache_enabled() if enabled is None else enabled):
        return
    write_stage_cache(path, metadata=metadata, payload=payload)


def cached_product_from_record(record: dict[str, Any]) -> NormalizedProduct:
    """Rehydrate a normalized product from a cache record."""
    payload = dict(record)
    payload["model_tokens"] = tuple(payload.get("model_tokens") or [])
    payload["quality_flags"] = tuple(payload.get("quality_flags") or [])
    payload["identifiers"] = dict(payload.get("identifiers") or {})
    payload.pop("extracted_attributes", None)
    return NormalizedProduct(**payload)


def read_normalization_cache(path: Path) -> list[NormalizedProduct] | None:
    """Read cached normalized products from disk."""
    blob = read_stage_cache(path)
    if blob is None:
        return None
    rows = blob["payload"].get("products")
    if not isinstance(rows, list):
        return None
    try:
        return [cached_product_from_record(item) for item in rows if isinstance(item, dict)]
    except (TypeError, ValueError):
        return None


def write_normalization_cache(path: Path, *, products: list[NormalizedProduct], metadata: dict[str, Any]) -> None:
    """Write cached normalized products to disk."""
    write_stage_cache(
        path,
        metadata=metadata,
        payload={"products": [asdict(product) for product in products]},
    )


def read_embedding_cache(path: Path) -> dict[str, dict[str, Any]] | None:
    """Read a product embedding cache payload."""
    blob = read_stage_cache(path)
    if blob is None:
        return None
    rows = blob["payload"].get("entries")
    if not isinstance(rows, list):
        return None
    entries: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            if not isinstance(row, dict):
                continue
            source_id = str(row["source_id"])
            text_hash = str(row["text_hash"])
            embedding = [float(value) for value in row["embedding"]]
            entries[source_id] = {
                "text_hash": text_hash,
                "embedding": embedding,
            }
    except (KeyError, TypeError, ValueError):
        return None
    return entries


def write_embedding_cache(path: Path, *, entries: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Write a product embedding cache payload."""
    records = [
        {
            "source_id": source_id,
            "text_hash": str(entry["text_hash"]),
            "embedding": [float(value) for value in entry["embedding"]],
        }
        for source_id, entry in sorted(entries.items())
    ]
    write_stage_cache(
        path,
        metadata=metadata,
        payload={"entries": records},
    )
