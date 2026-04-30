# Cartsy Product Deduplication Pipeline

This project implements a product entity-resolution pipeline for the Cartsy coding challenge. It ingests messy retailer product exports, normalizes product fields, retrieves duplicate candidates through Postgres, reranks them with OpenAI-assisted semantic and structured signals, clusters high-confidence matches into `dedupe_id` groups, and writes inspectable output files.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start Postgres with pgvector:

```bash
docker compose up -d postgres
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Run a smoke pipeline:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products_202604290549_first20.csv \
  --output outputs
```

Run on the full local CSV:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products_202604290549.csv \
  --output outputs \
  --merge-threshold 0.84 \
  --near-miss-threshold 0.70
```

For a full uncapped run, pass `--max-candidate-pairs none`:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products_202604290549.csv \
  --output outputs \
  --merge-threshold 0.84 \
  --near-miss-threshold 0.70 \
  --max-block-size 5000 \
  --max-candidate-pairs none
```

Pipeline runs write to a timestamped run directory under the output directory, so `--output outputs` writes artifacts to a path like `outputs/run_20260430_150405`. The generated `run_id` and full output path are also recorded in `summary_report.json`.

The full raw CSV is intentionally ignored by git because it is large.

## Architecture

The pipeline implements the full staged architecture in `info/dedupe_architecture.md`:

- Ingestion reads the CSV through DuckDB with all source columns preserved as retailer offers.
- Normalization cleans text, parses JSON-like `description` and `specs`, and extracts deterministic signals: identifiers, sizes, pack counts, model tokens, and quality flags.
- Postgres stores normalized products, exact keys, extracted attributes, full-text vectors, trigram indexes, and pgvector embeddings.
- Exact retrieval joins global identifiers, marketplace IDs, retailer SKU, and canonical URL keys.
- Lexical retrieval uses Postgres full-text search over weighted brand, title, category, specs, and description text.
- Fuzzy retrieval uses `pg_trgm` title similarity within normalized brands.
- Semantic retrieval embeds unresolved product text with OpenAI `text-embedding-3-small` by default and retrieves neighbors with pgvector cosine distance.
- Attribute extraction uses OpenAI structured outputs with `gpt-5.4-nano` by default for candidate products that need pairwise clarification. Open-ended variant attributes like color, scent, flavor, material, and variant name live here, not in static normalization dictionaries.
- Scoring combines deterministic rule evidence with exact, FTS, trigram, vector, and LLM attribute signals.
- Clustering unions accepted duplicate pairs into canonical product groups, with cluster-level guards against conflicting brands, global identifiers, deterministic sizes, and LLM-extracted variant attributes.
- Reporting writes assignments, group records, near-miss diagnostic pairs, candidate pairs, normalized products, and a summary report.

Model names are configurable through `.env` because evaluator accounts may expose different OpenAI models:

```text
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EXTRACTION_MODEL=gpt-5.4-nano
CARTSY_LLM_EXTRACTION_LIMIT=100
```

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

`summary_report.json` includes runtime and OpenAI accounting metadata under `metrics`: total elapsed time, average seconds per input record, per-stage timing averages, token usage by model, and estimated OpenAI cost in USD.

If `polars` is unavailable, the pipeline falls back to CSV for parquet-style outputs and writes a small fallback marker.

Query a completed run:

```bash
.venv/bin/cartsy-dedupe search "cetaphil hidratante" --run outputs/run_20260430_150405 --limit 5
.venv/bin/cartsy-dedupe search "cetaphil moisturizing lotion" --run outputs/run_20260430_150405 --backend postgres --limit 5
.venv/bin/cartsy-dedupe index-artifacts --run outputs/run_20260430_150405
.venv/bin/cartsy-dedupe search-artifacts "similar maybelline matte lipstick but different shade" --run-id run_20260430_150405 --type near_miss --limit 5
.venv/bin/cartsy-dedupe group <dedupe_id> --run outputs/run_20260430_150405
.venv/bin/cartsy-dedupe explain <source_id_a> <source_id_b> --run outputs/run_20260430_150405
```

Search defaults to `--backend auto`: it tries the live Postgres tables first, using exact name/SKU checks, weighted full-text search, pg_trgm title similarity, and pgvector cosine search when `OPENAI_API_KEY` is available for the query embedding. If Postgres is not running, it falls back to the exported `product_assignments.csv` fuzzy search so saved run artifacts remain portable.

`index-artifacts` builds a separate Postgres/pgvector index over completed-run artifacts without replacing the file artifacts as source of truth. It creates searchable documents for dedupe groups, source offers, merge/near-miss pair evidence, and the run summary. `search-artifacts` can then retrieve graph-aware results with metadata links like `group:<dedupe_id>`, `offer:<source_id>`, and pair endpoints. Use `--no-embeddings` for lexical-only indexing; with `OPENAI_API_KEY`, both indexing and queries include semantic vectors.

## REST API

Serve completed run artifacts with:

```bash
.venv/bin/cartsy-dedupe serve --runs-root outputs --host 127.0.0.1 --port 8000
```

Useful endpoints:

```text
GET /health
GET /runs
GET /runs/{run_id}/summary
GET /runs/{run_id}/products?q=cetaphil&brand=cetaphil&retailer=amazon_br&min_confidence=0.9
GET /runs/{run_id}/search?q=cetaphil%20hidratante&backend=artifacts
GET /runs/{run_id}/artifact-search?q=similar%20lipstick&type=near_miss
GET /runs/{run_id}/groups/{dedupe_id}
GET /runs/{run_id}/explain?source_id_a=123&source_id_b=456
```

The product list endpoint supports filtering by query text, retailer, brand, dedupe ID, decision, minimum cluster confidence, limit, and offset. Product search shares the CLI search backend options: `auto`, `postgres`, and `artifacts`.

## Deduplication Strategy

The system uses conservative, explainable entity resolution with staged escalation:

- Strong positive evidence: matching EAN/GTIN/UPC, matching ASIN, same brand plus same title and compatible deterministic size/pack signals.
- Strong negative evidence: conflicting strong brands, conflicting global identifiers, incompatible model tokens, clearly incompatible deterministic sizes, or conflicting LLM-extracted variant attributes.
- Retrieval evidence: exact keys are strongest, FTS/trigram broaden the candidate set, and pgvector catches semantically similar wording.
- LLM evidence: structured extraction clarifies ambiguous variant fields and helps distinguish exact duplicate from same parent product line with different variant.
- Binary decision: a pair merges only when it clears `--merge-threshold` and has no hard contradiction. Pairs below that threshold do not merge.
- Near-miss diagnostics: pairs above `--near-miss-threshold` but below the merge threshold are written for analysis, not for a required human review workflow.

This favors avoiding false-positive merges, because a bad merge could attach the wrong shopping link to creator content.

## Confidence Diagnostics

`summary_report.json` includes threshold and explanation diagnostics:

- `candidate_pairs_scored`: all generated candidate pairs evaluated by the scorer.
- `candidate_pairs_kept`: merged pairs plus near-miss diagnostic pairs above `--near-miss-threshold`.
- `merged_pairs`: pairs that cleared `--merge-threshold` and had no hard contradiction.
- `near_miss_pairs`: plausible pairs below the merge threshold.
- `threshold_sensitivity`: how many kept pairs would merge at nearby thresholds.
- `decision_reason_counts`: top explanation signals for merged and non-merged near-miss pairs.
- `blocking`: candidate retrieval counts split across exact, FTS, trigram, vector, and OpenAI stages.
- `clustering`: accepted merge edges and merge edges blocked by the cluster guard.
- `metrics`: end-to-end runtime, average time per input record, stage timings, OpenAI token usage, and estimated costs.

## With More Time

- Persist embedding and extraction caches across runs instead of rebuilding run tables from scratch.
- Add source-specific scraper adapters and feed live catalog data into the same Postgres schema.
- Add labeled-pair calibration and train a small classifier over the current explainable feature vector.
- Add an optional review UI over `near_miss_pairs.csv`.
