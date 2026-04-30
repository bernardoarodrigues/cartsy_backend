from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cartsy_dedupe.config import PipelineConfig
from cartsy_dedupe.schemas import CandidatePair, NormalizedProduct

CACHE_SCHEMA_VERSION = 2


def pipeline_cache_root() -> Path:
    base = os.getenv("CARTSY_PIPELINE_CACHE_DIR")
    if base:
        return Path(base)
    return Path(".cache") / "cartsy-dedupe"


def normalization_cache_dir() -> Path:
    return pipeline_cache_root() / "normalization"


def stage_cache_dir(stage_name: str) -> Path:
    return pipeline_cache_root() / stage_name


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint(*relative_paths: str) -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    return {
        relative_path: file_sha256(package_root / relative_path)
        for relative_path in relative_paths
    }


def normalize_module_hash() -> str:
    return file_sha256(Path(__file__).resolve().parents[1] / "normalize.py")


def stage_env_fingerprint(names: list[str]) -> dict[str, str | None]:
    return {name: os.getenv(name) for name in names}


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalization_cache_key(*, input_path: Path, limit: int | None, normalize_hash: str) -> str:
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


def scoring_cache_key(
    *,
    retrieval_key: str,
    config: PipelineConfig,
    code: dict[str, str],
) -> str:
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
    return cache_key(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stage": "cluster",
            "scoring_key": scoring_key,
            "code": code,
        }
    )


def cache_path_for(stage_name: str, key: str) -> Path:
    return stage_cache_dir(stage_name) / f"{key}.json"


def product_signature(products: list[NormalizedProduct]) -> str:
    payload = [asdict(product) for product in products]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def pair_blocks_to_records(pair_blocks: dict[tuple[int, int], set[str]]) -> list[dict[str, object]]:
    return [
        {
            "left_index": left,
            "right_index": right,
            "blocking_keys": sorted(block_keys),
        }
        for (left, right), block_keys in sorted(pair_blocks.items())
    ]


def pair_blocks_from_records(records: list[dict[str, Any]]) -> dict[tuple[int, int], set[str]]:
    return {
        (int(record["left_index"]), int(record["right_index"])): set(record.get("blocking_keys") or [])
        for record in records
        if isinstance(record, dict)
    }


def candidate_pairs_to_records(candidate_pairs: list[CandidatePair]) -> list[dict[str, Any]]:
    return [asdict(pair) for pair in candidate_pairs]


def candidate_pairs_from_records(records: list[dict[str, Any]]) -> list[CandidatePair]:
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
    return json.loads(json.dumps(clusters, ensure_ascii=False, sort_keys=True))


def write_stage_cache(
    path: Path,
    *,
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "metadata": metadata,
        "payload": payload,
    }
    path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")


def read_stage_cache(path: Path) -> dict[str, Any] | None:
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


def cached_product_from_record(record: dict[str, Any]) -> NormalizedProduct:
    payload = dict(record)
    payload["model_tokens"] = tuple(payload.get("model_tokens") or [])
    payload["quality_flags"] = tuple(payload.get("quality_flags") or [])
    payload["identifiers"] = dict(payload.get("identifiers") or {})
    payload["extracted_attributes"] = dict(payload.get("extracted_attributes") or {})
    return NormalizedProduct(**payload)


def read_normalization_cache(path: Path) -> list[NormalizedProduct] | None:
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
    write_stage_cache(
        path,
        metadata=metadata,
        payload={"products": [asdict(product) for product in products]},
    )
