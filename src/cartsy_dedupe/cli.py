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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartsy-dedupe",
        description="Run the Cartsy product deduplication pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Ingest, normalize, dedupe, cluster, and report.")
    run.add_argument("--input", required=True, help="Input product CSV path.")
    run.add_argument(
        "--output",
        default="outputs",
        help="Output directory root. Run artifacts go under run_<timestamp>/ unless path already starts with run_.",
    )
    run.add_argument("--merge-threshold", type=float, default=0.84)
    run.add_argument("--near-miss-threshold", type=float, default=0.70)
    run.add_argument(
        "--max-block-size",
        type=parse_optional_int,
        default=None,
        help="Optional max block size (default: none).",
    )
    run.add_argument(
        "--max-candidate-pairs",
        type=parse_optional_int,
        default=None,
        help="Maximum candidate pairs to collect. Omit or use none/null/unlimited for uncapped.",
    )
    run.add_argument("--near-miss-limit", type=int, default=25_000)
    run.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = PipelineConfig(
            merge_threshold=args.merge_threshold,
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
