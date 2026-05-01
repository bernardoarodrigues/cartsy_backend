# Cartsy Product Deduplication Pipeline

Production-shaped product entity resolution for the Cartsy challenge. The current pipeline retrieves candidate product pairs with Postgres exact/FTS/trigram/vector layers, computes a stable pairwise ML feature vector, adds dense OpenAI semantic similarity for every scored candidate pair, and uses a trained logistic-regression model to decide merge/no-merge.

The canonical pipeline walkthrough is in `PIPELINE.md`.

## Setup From Scratch

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
cp .env.example .env
docker compose up -d postgres
```

Set `OPENAI_API_KEY` in `.env`. The pipeline requires OpenAI embeddings because `semantic_sim` is a dense feature for all scored candidate pairs.

## Train The Logistic Model

Start from labeled products and ground truth. The paths below use the sibling experiment checkout as the source data; replace them with any CSVs that follow the same product and `source_id,deduped_id` schema.

```bash
.venv/bin/cartsy-dedupe augment-training-data \
  --input /Users/bernardorodrigues/Documents/Code/cartsy/data/dataset_v1.csv \
  --ground-truth /Users/bernardorodrigues/Documents/Code/cartsy/data/ground_truth_v1.csv \
  --output-data data/dataset_v1_augmented.csv \
  --output-ground-truth data/ground_truth_v1_augmented.csv \
  --output-manifest data/augmentation_manifest.csv \
  --duplicate-samples 1000 \
  --hard-negative-samples 300

.venv/bin/cartsy-dedupe train-model \
  --products data/dataset_v1_augmented.csv \
  --ground-truth data/ground_truth_v1_augmented.csv \
  --output-dir models \
  --use-openai-embeddings
```

Training writes:

```text
models/cartsy_logreg.joblib
models/metrics.json
models/threshold_curve.csv
models/feature_coefficients.csv
models/false_positives.csv
models/false_negatives.csv
models/top_risky_clusters.csv
```

Set `CARTSY_ML_MODEL_PATH=models/cartsy_logreg.joblib` in `.env`, or pass `--ml-model`.

## Run Deduplication

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products.csv \
  --output outputs \
  --ml-model models/cartsy_logreg.joblib \
  --merge-threshold 0.84 \
  --near-miss-threshold 0.70
```

Use `--dev` for progress bars and stage logs. Use `--max-block-size none --max-candidate-pairs none` for uncapped validation when the machine and database can handle it.

Run artifacts go under timestamped directories such as `outputs/run_20260501_193000`.

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

`summary_report.json` includes candidate counts, merge counts, threshold sensitivity, clustering diagnostics, stage timings, OpenAI usage/cost estimates, and a `stage_caches.stage_caching.enabled=0` marker. Stage caching is deliberately disabled while the ML scorer is the source of truth; product embedding caching remains available to avoid repeated OpenAI calls for unchanged product text.

## Query Completed Runs

```bash
.venv/bin/cartsy-dedupe search "cetaphil hidratante" --run outputs/run_20260501_193000 --limit 5
.venv/bin/cartsy-dedupe group <dedupe_id> --run outputs/run_20260501_193000
.venv/bin/cartsy-dedupe explain <source_id_a> <source_id_b> --run outputs/run_20260501_193000
.venv/bin/cartsy-dedupe index-artifacts --run outputs/run_20260501_193000
.venv/bin/cartsy-dedupe search-artifacts "similar lipstick different shade" --run-id run_20260501_193000 --type near_miss
```

## REST API

```bash
.venv/bin/cartsy-dedupe serve --runs-root outputs --host 127.0.0.1 --port 8000
```

Useful endpoints:

```text
GET /health
GET /runs
GET /runs/{run_id}/summary
GET /runs/{run_id}/products?q=cetaphil
GET /runs/{run_id}/search?q=cetaphil%20hidratante&backend=artifacts
GET /runs/{run_id}/artifact-search?q=similar%20lipstick&type=near_miss
GET /runs/{run_id}/groups/{dedupe_id}
GET /runs/{run_id}/explain?source_id_a=123&source_id_b=456
```

## Tests

```bash
.venv/bin/python -m pytest -q
```
