from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from dotenv import load_dotenv

from cartsy_dedupe.utils.pipeline_cache import (
    embedding_cache_dir,
    normalization_cache_dir,
    read_normalization_cache,
)
from cartsy_dedupe.utils.pipeline_helpers import embedding_text, ensure_openai_api_key
from cartsy_dedupe.utils.pipeline_metrics import RunMetrics

try:  # pragma: no cover - import failure depends on local env setup.
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cartsy_dedupe.embed_cache",
        description=(
            "Generate OpenAI embeddings from the newest normalized cache file and "
            "save them for deterministic test use."
        ),
    )
    parser.add_argument(
        "--normalization-dir",
        default=None,
        help="Directory containing normalized cache JSON files. Default: pipeline normalization cache dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where embeddings and source-id index files will be written. "
            "Default: pipeline embedding cache dir (embeddings/all-products)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CARTSY_EMBEDDING_BATCH_SIZE", "128")),
        help="OpenAI embedding request batch size.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model passed to OpenAI.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Numpy dtype for output embedding matrix.",
    )
    return parser


def resolve_normalization_dir(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    fallback = normalization_cache_dir()
    if fallback.exists():
        return fallback
    raise RuntimeError(
        f"Normalization directory not found at '{preferred}' or fallback '{fallback}'."
    )


def latest_normalization_file(normalization_dir: Path) -> Path:
    candidates = sorted(
        normalization_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No normalization cache files found in '{normalization_dir}'.")
    return candidates[0]


def batched(items: list[Any], size: int) -> list[list[Any]]:
    safe_size = max(1, size)
    return [items[index : index + safe_size] for index in range(0, len(items), safe_size)]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_dotenv(dotenv_path=Path.cwd() / ".env")
    if OpenAI is None:
        raise RuntimeError("Install openai before generating embeddings.")
    ensure_openai_api_key()

    normalization_seed = Path(args.normalization_dir) if args.normalization_dir else normalization_cache_dir()
    normalization_dir = resolve_normalization_dir(normalization_seed)
    cache_file = latest_normalization_file(normalization_dir)
    products = read_normalization_cache(cache_file)
    if products is None:
        raise RuntimeError(f"Could not read normalized products from '{cache_file}'.")
    if not products:
        raise RuntimeError(f"Normalized cache file '{cache_file}' has zero products.")

    products = sorted(products, key=lambda product: product.source_id)
    texts = [
        embedding_text(
            brand=product.brand_raw,
            title=product.name_raw,
            category=product.category_raw,
            description=product.description_raw,
            specs=product.specs_raw,
            dimension=product.dimension_raw,
        )
        for product in products
    ]

    client = OpenAI()
    embedding_rows: list[list[float]] = []
    metrics = RunMetrics()
    started = perf_counter()
    batches = batched(texts, args.batch_size)
    for index, batch in enumerate(batches, start=1):
        response = client.embeddings.create(model=args.embedding_model, input=batch)
        metrics.add_usage(args.embedding_model, getattr(response, "usage", None))
        embedding_rows.extend([item.embedding for item in response.data])
        print(f"embedded batch {index}/{len(batches)} ({len(embedding_rows):,} products)")

    output_dir = Path(args.output_dir) if args.output_dir is not None else embedding_cache_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"embeddings_{cache_file.stem}_{timestamp}"
    matrix_path = output_dir / f"{stem}.npy"
    source_ids_path = output_dir / f"{stem}.source_ids.json"
    source_id_to_index_path = output_dir / f"{stem}.source_id_to_index.json"
    metrics_path = output_dir / f"{stem}.metrics.json"

    embedding_matrix = np.asarray(embedding_rows, dtype=np.float32 if args.dtype == "float32" else np.float64)
    np.save(matrix_path, embedding_matrix)

    source_ids = [product.source_id for product in products]
    source_id_to_index = {source_id: idx for idx, source_id in enumerate(source_ids)}
    source_ids_path.write_text(json.dumps(source_ids, ensure_ascii=False), encoding="utf-8")
    source_id_to_index_path.write_text(
        json.dumps(source_id_to_index, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    elapsed_seconds = perf_counter() - started
    metrics_report = metrics.as_report(
        embedding_model=args.embedding_model,
        extraction_model="not_used",
        input_records=len(source_ids),
        total_elapsed_seconds=elapsed_seconds,
    )
    total_cost_usd = float(metrics_report["openai"]["total_estimated_cost_usd"])
    estimated_cost_per_row_usd = total_cost_usd / len(source_ids) if source_ids else 0.0
    report = {
        "normalization_cache_path": str(cache_file),
        "products_embedded": len(source_ids),
        "batch_size": max(1, args.batch_size),
        "estimated_cost_per_row_usd": round(estimated_cost_per_row_usd, 10),
        "output_files": {
            "embeddings_npy": str(matrix_path),
            "source_ids_json": str(source_ids_path),
            "source_id_to_index_json": str(source_id_to_index_path),
            "metrics_json": str(metrics_path),
        },
        "metrics": metrics_report,
    }
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"normalization cache: {cache_file}")
    print(f"products embedded: {len(source_ids):,}")
    print(f"embedding model: {args.embedding_model}")
    print(f"embedding matrix: {matrix_path}")
    print(f"source ids: {source_ids_path}")
    print(f"source id index: {source_id_to_index_path}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
