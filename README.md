# Cartsy Product Deduplication Pipeline

![Deduped groups](diagrams/deduped-groups.jpeg)

Product entity resolution pipeline for the Cartsy challenge. The project ingests messy product CSVs, normalizes them into a stable schema, retrieves candidate duplicate pairs with Postgres exact/FTS/trigram/vector layers, scores uncertain pairs with a calibrated logistic-regression model, and writes queryable deduped-product artifacts.

## What Is Included

- `src/cartsy_dedupe/`: ingestion, normalization, retrieval, scoring, clustering, artifact search, API, and training code.
- `models/final_submission/`: committed final logistic-regression model plus training diagnostics.
- `PIPELINE.md`: pipeline walkthrough and runtime trade-offs
- `TRAINING.md`: supervised training, augmentation, threshold selection, and artifact walkthrough

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

- OpenAI: keep `CARTSY_EMBEDDING_PROVIDER=openai`, set `OPENAI_API_KEY`, and keep `CARTSY_EMBEDDING_DIMENSIONS=1536` for `text-embedding-3-small`.
- Local: set `CARTSY_EMBEDDING_PROVIDER=sentence-transformers`, usually with `CARTSY_EMBEDDING_DIMENSIONS=384` for `all-MiniLM-L6-v2`.

## Run The Pipeline

Place the full challenge CSV at `data/products.csv`, then run:

```bash
.venv/bin/cartsy-dedupe run \
  --input data/products.csv \
  --ml-model models/final_submission/cartsy_logreg.joblib \
  --merge-threshold 0.84 \
  --evidence-merge-threshold 0.78 \
  --near-miss-threshold 0.70 \
  --near-miss-limit 50000 \
  --max-block-size none \
  --max-candidate-pairs none \
  --dev
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

## Querying A Completed Run

REST API

```bash
.venv/bin/cartsy-dedupe serve --runs-root outputs --host 127.0.0.1 --port 8000
```

CLI

```bash
.venv/bin/cartsy-dedupe search "cetaphil hidratante" --run outputs/run_YYYYMMDD_HHMMSS --limit 5
.venv/bin/cartsy-dedupe group <dedupe_id> --run outputs/run_YYYYMMDD_HHMMSS
.venv/bin/cartsy-dedupe explain <source_id_a> <source_id_b> --run outputs/run_YYYYMMDD_HHMMSS
```

Semantic indexing over output artifacts:

```bash
.venv/bin/cartsy-dedupe index-artifacts --run outputs/run_YYYYMMDD_HHMMSS
.venv/bin/cartsy-dedupe search-artifacts "similar lipstick different shade" --run-id run_YYYYMMDD_HHMMSS --type near_miss
```

## Architecture Overview

See `PIPELINE.md` for the full dedupe pipeline details.

![Cartsy runtime dedupe pipeline](diagrams/dedupe-pipeline.svg)

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

See `TRAINING.md` for the full training pipeline details. The committed final model was trained with controlled positive augmentation, dirty-identifier hard negatives, embeddings, calibration, threshold curves, false-positive/false-negative exports, feature coefficients, and risky-cluster diagnostics.

![Cartsy supervised training pipeline](diagrams/training-pipeline.svg)

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

- Enhance accuracy by iterating on supervised training with more labeled pairs, harder negatives, and threshold calibration on held-out folds.
- Compare or combine other classifiers beyond logistic regression (for example XGBoost) on the same pairwise feature matrix and calibration gates.
- Add a small reviewer web UI for low-confidence merges and near misses.
- Add source-specific trust profiles so retailer SKU, marketplace URLs, and third-party catalog IDs can be weighted by source quality.
- Add active-learning loops from false positives/false negatives back into training data.
- Move full-run orchestration to a managed job runner when processing grows beyond one local Postgres instance.
- Add incremental ingestion so unchanged source rows reuse existing normalized records and embeddings across daily feeds.
