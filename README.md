# Cartsy Product Deduplication Pipeline

Production-shaped product entity resolution for the Cartsy product-data challenge. The project ingests messy product CSVs, normalizes them into a stable schema, retrieves candidate duplicate pairs with Postgres exact/FTS/trigram/vector layers, scores uncertain pairs with a calibrated logistic-regression model, and writes queryable deduped-product artifacts.

The implementation is intentionally CLI-first: reviewers can run the batch pipeline, inspect saved artifacts, query a completed run from the terminal, or start the optional read-only REST API.

## What Is Included

- `src/cartsy_dedupe/`: ingestion, normalization, retrieval, scoring, clustering, artifact search, API, and training code.
- `data/products_first20.csv`: tiny public smoke fixture. The full challenge CSVs are expected under `data/` but are not committed because they are large/local inputs.
- `models/final_submission/`: committed final logistic-regression model plus training diagnostics.
- `outputs/sample/`: small sample output snapshot from the final full-data run.
- `PIPELINE.md`: operational pipeline walkthrough and runtime trade-offs.
- `TRAINING.md`: supervised training, augmentation, threshold selection, and artifact guide.

## Quick Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres
```

`requirements.txt` installs the package in editable mode plus runtime/test dependencies. The default `.env.example` points at the committed final model:

```text
CARTSY_ML_MODEL_PATH=models/final_submission/cartsy_logreg.joblib
```

For full vector retrieval, choose one embedding backend:

- OpenAI: keep `CARTSY_EMBEDDING_PROVIDER=openai`, set `OPENAI_API_KEY`, and keep `CARTSY_EMBEDDING_DIMENSIONS=1536`.
- Local: set `CARTSY_EMBEDDING_PROVIDER=sentence-transformers`, usually with `CARTSY_EMBEDDING_DIMENSIONS=384` for `all-MiniLM-L6-v2`.

## No-Database Demo

The committed sample output can be queried without Postgres, OpenAI, or the full dataset:

```bash
.venv/bin/cartsy-dedupe search "wella" --run outputs/sample --backend artifacts --limit 5
.venv/bin/cartsy-dedupe group prod_da4b9237bacc --run outputs/sample
```

Sample full-run summary snapshot:

```text
input_records=246,969
candidate_pairs_scored=3,161,740
candidate_pairs_kept=145,170
merged_pairs=5,044
final_unique_products=245,434
duplicate_records_grouped=1,535
```

## Run The Pipeline

Place the full challenge CSV at `data/products.csv`, then run:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products.csv \
  --output outputs \
  --ml-model models/final_submission/cartsy_logreg.joblib \
  --merge-threshold 0.84 \
  --evidence-merge-threshold 0.78 \
  --near-miss-threshold 0.70
```

Run artifacts are written under timestamped directories such as `outputs/run_20260503_130136`:

```text
normalized_products.parquet
candidate_pairs.parquet
product_assignments.csv
dedupe_groups.jsonl
near_miss_pairs.csv
summary_report.json
```

Use `--dev` for progress logs. Use `--max-block-size none --max-candidate-pairs none` only for uncapped validation on a machine that can handle the larger candidate set.

## Query A Completed Run

```bash
.venv/bin/cartsy-dedupe search "cetaphil hidratante" --run outputs/run_YYYYMMDD_HHMMSS --limit 5
.venv/bin/cartsy-dedupe group <dedupe_id> --run outputs/run_YYYYMMDD_HHMMSS
.venv/bin/cartsy-dedupe explain <source_id_a> <source_id_b> --run outputs/run_YYYYMMDD_HHMMSS
```

Optional semantic indexing over completed artifacts:

```bash
.venv/bin/cartsy-dedupe index-artifacts --run outputs/run_YYYYMMDD_HHMMSS
.venv/bin/cartsy-dedupe search-artifacts "similar lipstick different shade" --run-id run_YYYYMMDD_HHMMSS --type near_miss
```

Optional REST API:

```bash
.venv/bin/cartsy-dedupe serve --runs-root outputs --host 127.0.0.1 --port 8000
```

Useful endpoints include `/health`, `/runs`, `/runs/{run_id}/summary`, `/runs/{run_id}/products`, `/runs/{run_id}/groups/{dedupe_id}`, and `/runs/{run_id}/explain`.

## Architecture Overview

The pipeline follows a conservative entity-resolution shape:

1. Ingest CSV rows with tolerant parsing.
2. Normalize product fields into `NormalizedProduct`: source id, retailer, names, brand, category, price, size, pack count, model tokens, identifiers, and quality flags.
3. Load normalized rows into Postgres working tables.
4. Retrieve candidate pairs with exact keys, FTS, trigram title similarity, and evidence-gated pgvector search.
5. Build pairwise features in `src/cartsy_dedupe/features.py`.
6. Apply deterministic rule guards for certain matches and hard contradictions.
7. Score uncertain pairs with the calibrated logistic-regression model.
8. Require both model probability and independent evidence for non-rule ML merges.
9. Cluster accepted merge edges with connected components plus cluster-level contradiction guards.
10. Write artifacts and a summary report with metrics, cache status, quality flags, and low-confidence diagnostics.

The core trade-off is precision over aggressive grouping. Exact identifiers and trusted product URLs can fast-path obvious duplicates, but variants with conflicting size, shade, pack, kit/component, or form evidence are blocked or penalized even when text similarity is high.

## Deduplication Strategy

Two products are treated as the same purchasable item only when enough independent evidence agrees:

- Exact evidence: shared EAN/GTIN/UPC, ASIN, same-retailer SKU, or trusted canonical product URL.
- Lexical evidence: brand-aware FTS and trigram title similarity.
- Semantic evidence: dense embedding cosine similarity for candidate-pair products.
- Structured evidence: size, pack count, model tokens, identifiers, price ratio, category, and variant/kit/form contradiction features.
- ML evidence: calibrated logistic regression trained on pairwise labels and hard negatives.

`CERTAIN_BLOCK` rules override the model for factual contradictions. `CERTAIN_MATCH` rules bypass ML only for high-trust exact matches. Everything in between is scored by the model and must also pass `--evidence-merge-threshold`, which is the guard against sparse vector-only overconfidence.

## Training

See `TRAINING.md` for the full training pipeline. The committed final model was trained with controlled positive augmentation, dirty-identifier hard negatives, embeddings, calibration, threshold curves, false-positive/false-negative exports, feature coefficients, and risky-cluster diagnostics.

## Evaluation

```bash
.venv/bin/cartsy-dedupe evaluate-run \
  --run outputs/run_YYYYMMDD_HHMMSS \
  --ground-truth data/ground_truth_merged.csv \
  --min-precision 0.97 \
  --min-recall 0.80 \
  --min-vector-only-precision 0.95
```

The evaluator writes `labeled_evaluation.json`. Blank `deduped_id` labels are ignored by default so accidental blank-label clusters do not inflate positives.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests cover normalization, rule scoring, pairwise features, clustering, cache keys, artifact queries, API behavior, training helpers, and labeled evaluation.

## What I Would Improve Next

- Add a small reviewer web UI for low-confidence merges and near misses.
- Add source-specific trust profiles so retailer SKU, marketplace URLs, and third-party catalog IDs can be weighted by source quality.
- Add active-learning loops from false positives/false negatives back into training data.
- Move full-run orchestration to a managed job runner when processing grows beyond one local Postgres instance.
- Add incremental ingestion so unchanged source rows reuse existing normalized records and embeddings across daily feeds.
