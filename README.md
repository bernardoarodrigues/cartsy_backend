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

Train from the merged 68k labels plus controlled augmentation. The 68k label file is still highly imbalanced, so augmentation should add guarded positive duplicates and a smaller set of dirty-identifier hard negatives rather than simply adding more singleton negatives.

```bash
.venv/bin/cartsy-dedupe augment-training-data \
  --input data/products.csv \
  --ground-truth data/ground_truth_merged.csv \
  --output-data data/dataset_merged_augmented.csv \
  --output-ground-truth data/ground_truth_merged_augmented.csv \
  --output-manifest data/augmentation_manifest_merged.csv \
  --duplicate-samples 5000 \
  --hard-negative-samples 1000
```

```bash
.venv/bin/cartsy-dedupe train-model \
  --products data/dataset_merged_augmented.csv \
  --ground-truth data/ground_truth_merged_augmented.csv \
  --output-dir models \
  --target-precision 0.97 \
  --min-recall 0.50 \
  --cv-folds 5 \
  --max-positive-pairs 20000 \
  --max-hard-negative-pairs 60000 \
  --use-embeddings
```

The model threshold is precision-constrained, with a recall guard. If the only thresholds satisfying the precision floor merge almost nothing, training records that the floor was unmet and uses the best-F1 operating point instead. Ties prefer the lower threshold that preserves the same precision/recall/F1 so calibrated plateaus do not become needlessly strict `0.99` cutoffs.

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
  --evidence-merge-threshold 0.70 \
  --near-miss-threshold 0.70
```

Use `--dev` for progress bars and stage logs. Use `--max-block-size none --max-candidate-pairs none` for uncapped validation when the machine and database can handle it.

Run artifacts go under timestamped directories such as `outputs/run_20260501_193000`.

Evaluate completed runs against labels before trusting a model:

```bash
.venv/bin/cartsy-dedupe evaluate-run \
  --run outputs/run_20260501_193000 \
  --ground-truth data/ground_truth_merged.csv
```

This writes `labeled_evaluation.json` in the run directory with overall precision/recall plus risky slices such as vector-only and generic-brand candidate pairs. Blank `deduped_id` labels are ignored by default so accidental empty-label clusters do not inflate positives.

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
Pair artifacts separate `ml_score`, `evidence_score`, `decision_threshold`, and `decision_reason`; `score` is the evidence confidence used for display and cluster confidence, not a policy-clamped copy of the model threshold. Non-rule ML merges require both `ml_score >= decision_threshold` and `evidence_score >= --evidence-merge-threshold`, which prevents sparse vector-only candidates from merging solely because the model is overconfident.

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
