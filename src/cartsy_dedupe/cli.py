from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    run.add_argument("--output", required=True, help="Output directory.")
    run.add_argument("--merge-threshold", type=float, default=0.84)
    run.add_argument("--near-miss-threshold", type=float, default=0.70)
    run.add_argument("--max-block-size", type=int, default=1_500)
    run.add_argument("--max-candidate-pairs", type=int, default=2_000_000)
    run.add_argument("--near-miss-limit", type=int, default=25_000)
    run.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")

    search = subparsers.add_parser("search", help="Search product assignments in a completed run.")
    search.add_argument("query", help="Search text.")
    search.add_argument("--run", required=True, help="Run output directory.")
    search.add_argument("--limit", type=int, default=10)
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
        report = run_pipeline(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            config=config,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "search":
        results = search_products(args.run, args.query, limit=args.limit)
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
