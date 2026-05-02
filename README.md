# Cartsy Product Deduplication Pipeline

Production-shaped product entity resolution for the Cartsy challenge. The pipeline retrieves candidate product pairs with Postgres exact/FTS/trigram/vector layers, evaluates each pair through a condition-based certainty chain (hard-blocking contradictions and fast-pathing exact identifier matches), and uses a calibrated logistic-regression model with cross-validated threshold selection for all uncertain cases.

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

Choose an embedding backend in `.env`: `CARTSY_EMBEDDING_PROVIDER=openai` uses OpenAI and requires `OPENAI_API_KEY`; `CARTSY_EMBEDDING_PROVIDER=sentence-transformers` runs local sentence-transformers embeddings. Keep `CARTSY_EMBEDDING_DIMENSIONS` aligned with the model because pgvector columns have fixed width.

## Train The Logistic Model

Train from the augmented experiment dataset. The large augmented CSVs are local inputs and are intentionally ignored by git. Place your augmented dataset CSVs at `data/dataset_v1_augmented.csv` and `data/ground_truth_v1_augmented.csv` (or adjust the paths in the command below).

```bash
.venv/bin/cartsy-dedupe train-model \
  --products data/dataset_v1_augmented.csv \
  --ground-truth data/ground_truth_v1_augmented.csv \
  --output-dir models \
  --cv-folds 5 \
  --max-positive-pairs 10000 \
  --max-hard-negative-pairs 30000 \
  --use-embeddings
```

To regenerate an augmented dataset from base labels, use `cartsy-dedupe augment-training-data`; the pipeline runbook in `PIPELINE.md` describes the positive and hard-negative patterns.

Training writes:

```text
models/cartsy_logreg.joblib
models/metrics.json
models/threshold_curve.csv
models/calibration_threshold_curve.csv
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

`summary_report.json` includes candidate counts, merge counts, threshold sensitivity, clustering diagnostics, stage timings, embedding usage/cost estimates when using OpenAI, and per-stage cache paths for debugging. Product embedding caching is available to avoid recomputing embeddings for unchanged products across repeated runs.

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
