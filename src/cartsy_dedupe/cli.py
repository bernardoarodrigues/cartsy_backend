from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .artifact_index import index_artifacts, search_artifacts
from .config import PipelineConfig
from .pipeline import run_pipeline
from .query import explain_pair, get_group, print_table, search_products
from .training import augment_training_data, train_logistic_regression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartsy-dedupe",
        description="Run the Cartsy product deduplication pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Ingest, normalize, dedupe, cluster, and report.")
    run.add_argument("--input", default="data/products.csv", help="Input product CSV path.")
    run.add_argument(
        "--output",
        default="outputs",
        help="Output directory root. Run artifacts go under run_<timestamp>/ unless path already starts with run_.",
    )
    run.add_argument("--merge-threshold", type=float, default=0.84)
    run.add_argument(
        "--ml-model",
        default=os.getenv("CARTSY_ML_MODEL_PATH"),
        help="Path to a cartsy_logreg.joblib bundle from `cartsy-dedupe train-model`.",
    )
    run.add_argument("--near-miss-threshold", type=float, default=0.70)
    run.add_argument(
        "--max-block-size",
        type=parse_optional_int,
        default=PipelineConfig.max_block_size,
        help=f"Optional max block size (default: {PipelineConfig.max_block_size}).",
    )
    run.add_argument(
        "--max-candidate-pairs",
        type=parse_optional_int,
        default=PipelineConfig.max_candidate_pairs,
        help=(
            "Maximum candidate pairs to collect "
            f"(default: {PipelineConfig.max_candidate_pairs:,}). "
            "Use none/null/unlimited for uncapped."
        ),
    )
    run.add_argument("--near-miss-limit", type=int, default=25_000)
    run.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    run.add_argument(
        "--dev",
        action="store_true",
        help="Print per-stage debug logs and show progress bars during pipeline execution.",
    )

    search = subparsers.add_parser("search", help="Search product assignments in a completed run.")
    search.add_argument("query", help="Search text.")
    search.add_argument("--run", required=True, help="Run output directory.")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--backend",
        choices=["auto", "postgres", "artifacts"],
        default="auto",
        help="Search backend. auto tries Postgres/pgvector first and falls back to run artifacts.",
    )
    search.add_argument("--json", action="store_true", help="Print JSON instead of a table.")

    group = subparsers.add_parser("group", help="Show a dedupe group and its source offers.")
    group.add_argument("dedupe_id", help="Dedupe group ID.")
    group.add_argument("--run", required=True, help="Run output directory.")
    group.add_argument("--json", action="store_true", help="Print JSON instead of a table.")

    explain = subparsers.add_parser("explain", help="Explain a candidate pair from a completed run.")
    explain.add_argument("source_id_a")
    explain.add_argument("source_id_b")
    explain.add_argument("--run", required=True, help="Run output directory.")
    explain.add_argument("--json", action="store_true", help="Print JSON instead of a readable summary.")

    index = subparsers.add_parser("index-artifacts", help="Index completed run artifacts into Postgres/pgvector.")
    index.add_argument("--run", required=True, help="Run output directory.")
    index.add_argument("--run-id", default=None, help="Stable run ID for indexed artifacts. Defaults to run directory name.")
    index.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")
    index.add_argument("--no-embeddings", action="store_true", help="Index lexical metadata only, without OpenAI embeddings.")

    artifact_search = subparsers.add_parser("search-artifacts", help="Semantic search across indexed groups, offers, pairs, and summaries.")
    artifact_search.add_argument("query", help="Search text.")
    artifact_search.add_argument("--run-id", default=None, help="Optional indexed run ID filter.")
    artifact_search.add_argument("--type", choices=["group", "offer", "pair", "near_miss", "summary"], default=None)
    artifact_search.add_argument("--limit", type=int, default=10)
    artifact_search.add_argument("--json", action="store_true", help="Print JSON instead of a table.")

    serve = subparsers.add_parser("serve", help="Run the REST API over completed run artifacts.")
    serve.add_argument("--runs-root", default="outputs", help="Directory containing run_* output folders.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    augment = subparsers.add_parser("augment-training-data", help="Create guarded synthetic positives and dirty-identifier hard negatives.")
    augment.add_argument("--input", required=True, help="Input product CSV path.")
    augment.add_argument("--ground-truth", required=True, help="Ground-truth CSV with source_id,deduped_id.")
    augment.add_argument("--output-data", required=True, help="Augmented product CSV path.")
    augment.add_argument("--output-ground-truth", required=True, help="Augmented ground-truth CSV path.")
    augment.add_argument("--output-manifest", required=True, help="Augmentation manifest CSV path.")
    augment.add_argument("--duplicate-samples", type=int, required=True)
    augment.add_argument("--hard-negative-samples", type=int, default=None)
    augment.add_argument("--start-source-id", type=int, default=500_000)
    augment.add_argument("--start-deduped-id", type=int, default=500_000)
    augment.add_argument("--seed", type=int, default=7)

    train = subparsers.add_parser("train-model", help="Train the logistic-regression pair scorer and write eval artifacts.")
    train.add_argument("--products", required=True, help="Training product CSV path.")
    train.add_argument("--ground-truth", required=True, help="Ground-truth CSV with source_id,deduped_id.")
    train.add_argument("--output-dir", required=True, help="Directory for model and eval artifacts.")
    train.add_argument("--target-precision", type=float, default=0.97)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--max-positive-pairs", type=int, default=50_000)
    train.add_argument("--max-hard-negative-pairs", type=int, default=150_000)
    train.add_argument("--use-openai-embeddings", action="store_true", help="Compute dense semantic_sim using OpenAI embeddings during training.")
    train.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = PipelineConfig(
            merge_threshold=args.merge_threshold,
            ml_model_path=args.ml_model,
            near_miss_threshold=args.near_miss_threshold,
            max_block_size=args.max_block_size,
            max_candidate_pairs=args.max_candidate_pairs,
            near_miss_limit=args.near_miss_limit,
        )
        try:
            report = run_pipeline(
                input_path=Path(args.input),
                output_dir=resolve_run_output_dir(Path(args.output)),
                config=config,
                limit=args.limit,
                dev=args.dev,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "search":
        try:
            results = search_products(args.run, args.query, limit=args.limit, backend=args.backend)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print_table(
                results,
                ["score", "source_id", "dedupe_id", "retailer", "brand", "price_cents", "name"],
            )
        return 0

    if args.command == "group":
        group = get_group(args.run, args.dedupe_id)
        if args.json:
            print(json.dumps(group, indent=2, ensure_ascii=False))
        else:
            print(json.dumps({key: value for key, value in group.items() if key != "offers"}, indent=2, ensure_ascii=False))
            print()
            print_table(
                list(group.get("offers", [])),
                ["source_id", "retailer", "brand", "price_cents", "dimension", "name"],
            )
        return 0

    if args.command == "explain":
        explanation = explain_pair(args.run, args.source_id_a, args.source_id_b)
        if args.json:
            print(json.dumps(explanation, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(explanation, indent=2, ensure_ascii=False))
        return 0

    if args.command == "index-artifacts":
        try:
            report = index_artifacts(
                args.run,
                run_id=args.run_id,
                batch_size=args.batch_size,
                embed=not args.no_embeddings,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "search-artifacts":
        try:
            results = search_artifacts(args.query, run_id=args.run_id, limit=args.limit, artifact_type=args.type)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print_table(results, ["score", "run_id", "artifact_type", "artifact_id", "title"])
        return 0

    if args.command == "serve":
        try:
            import uvicorn

            from .api import create_app
        except ImportError as exc:
            print("error: install FastAPI dependencies with `pip install -r requirements.txt`", file=sys.stderr)
            return 1
        os.environ["CARTSY_RUNS_ROOT"] = args.runs_root
        app = "cartsy_dedupe.api:create_app" if args.reload else create_app(runs_root=args.runs_root)
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, factory=args.reload)
        return 0

    if args.command == "augment-training-data":
        try:
            report = augment_training_data(
                input_path=args.input,
                ground_truth_path=args.ground_truth,
                output_data_path=args.output_data,
                output_ground_truth_path=args.output_ground_truth,
                output_manifest_path=args.output_manifest,
                duplicate_samples=args.duplicate_samples,
                hard_negative_samples=args.hard_negative_samples,
                start_source_id=args.start_source_id,
                start_deduped_id=args.start_deduped_id,
                seed=args.seed,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "train-model":
        try:
            report = train_logistic_regression(
                products_path=args.products,
                ground_truth_path=args.ground_truth,
                output_dir=args.output_dir,
                target_precision=args.target_precision,
                random_state=args.random_state,
                max_positive_pairs=args.max_positive_pairs,
                max_hard_negative_pairs=args.max_hard_negative_pairs,
                use_openai_embeddings=args.use_openai_embeddings,
                embedding_model=args.embedding_model,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "unlimited", "uncapped"}:
        return None
    return int(value)


def resolve_run_output_dir(output_dir: Path, *, now: datetime | None = None) -> Path:
    if output_dir.name.startswith("run_"):
        return output_dir
    run_id = (now or datetime.now()).strftime("run_%Y%m%d_%H%M%S")
    return output_dir / run_id


if __name__ == "__main__":
    raise SystemExit(main())
