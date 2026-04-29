from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartsy-dedupe",
        description="Run the Cartsy product deduplication pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Ingest, normalize, dedupe, cluster, and report.")
    run.add_argument("--input", required=True, help="Input product CSV path.")
    run.add_argument("--output", required=True, help="Output directory.")
    run.add_argument("--auto-threshold", type=float, default=0.86)
    run.add_argument("--review-threshold", type=float, default=0.70)
    run.add_argument("--max-block-size", type=int, default=1_500)
    run.add_argument("--max-candidate-pairs", type=int, default=2_000_000)
    run.add_argument("--review-limit", type=int, default=25_000)
    run.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = PipelineConfig(
            auto_threshold=args.auto_threshold,
            review_threshold=args.review_threshold,
            max_block_size=args.max_block_size,
            max_candidate_pairs=args.max_candidate_pairs,
            review_limit=args.review_limit,
        )
        report = run_pipeline(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            config=config,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
