# Cartsy Product Deduplication Pipeline

This project implements a CLI-first product entity-resolution pipeline for the Cartsy coding challenge. It ingests messy retailer product exports, normalizes product fields, generates duplicate candidates, scores likely matches with explainable confidence, clusters high-confidence matches into `dedupe_id` groups, and writes inspectable output files.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Run the Layer 1 pipeline on the small fixture:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products_202604290549_first20.csv \
  --output outputs/run_layer1_first20
```

Run on the full local CSV:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products_202604290549.csv \
  --output outputs/run_full \
  --merge-threshold 0.84 \
  --near-miss-threshold 0.70
```

The full raw CSV is intentionally ignored by git because it is large.

## Architecture

The pipeline is organized as layers:

- Ingestion reads the CSV and preserves source rows as retailer offers.
- Normalization cleans text, parses JSON-like `description` and `specs`, extracts identifiers, sizes, model tokens, variant attributes, and quality flags.
- Blocking creates candidate pairs without comparing every row to every other row.
- Scoring assigns an explainable confidence score using brand, title, identifiers, model tokens, variant attributes, category, specs/description, and price.
- Clustering unions accepted duplicate pairs into canonical product groups.
- Reporting writes assignments, group records, near-miss diagnostic pairs, candidate pairs, normalized products, and a summary report.

The dedupe target is the same purchasable variant when variant attributes are clear. Size is a strong signal but not an unconditional blocker: missing or ambiguous size lowers confidence, while clearly incompatible sizes on both records prevent automatic merge.

## Outputs

Each run writes:

```text
normalized_products.parquet
candidate_pairs.parquet
product_assignments.csv
dedupe_groups.jsonl
near_miss_pairs.csv
summary_report.json
```

If `polars` is unavailable, the pipeline falls back to CSV for parquet-style outputs and writes a small fallback marker.

## Deduplication Strategy

The system uses conservative, explainable entity resolution:

- Strong positive evidence: matching EAN/GTIN/UPC, matching ASIN, same brand plus same title and compatible variant attributes.
- Strong negative evidence: conflicting strong brands, conflicting global identifiers, incompatible model tokens, clearly incompatible sizes, or clearly different variant attributes.
- Binary decision: a pair merges only when it clears `--merge-threshold` and has no hard contradiction. Pairs below that threshold do not merge.
- Near-miss diagnostics: pairs above `--near-miss-threshold` but below the merge threshold are written for analysis, not for a required human review workflow.

This favors avoiding false-positive merges, because a bad merge could attach the wrong shopping link to creator content.

## With More Time

- Add a richer brand alias dictionary from observed variants.
- Add category taxonomy normalization across Portuguese and English retailer paths.
- Add calibration/evaluation tooling for near-miss pairs if labeled examples become available.
- Add embeddings as an additional candidate-generation/scoring feature, while keeping identifiers and variant rules authoritative.
- Add a live scraper source as a bonus ingestion adapter.
