# Dedupe Pipeline

This is the production path implemented in `src/cartsy_dedupe/pipeline.py`.

## 1. Ingest And Normalize

`cartsy-dedupe run` reads the product CSV, normalizes every row, and loads the normalized products into Postgres. Normalization extracts stable deterministic signals:

- canonical text fields for brand, title, category, description, and specs
- global and marketplace identifiers such as EAN, GTIN, UPC, ASIN, SKU, and URL keys
- size, unit, pack count, model-like tokens, price, and quality flags

Trade-off: normalization stays deterministic and conservative. Open-ended attributes such as color, scent, flavor, and shade are not hardcoded into endless vocabularies.

## 2. Retrieve Candidate Pairs

Candidate generation is recall-oriented and layered:

- exact keys: shared global identifiers, marketplace identifiers, retailer SKU, and trusted canonical product URL keys
- FTS: weighted Postgres full-text search over brand, title, category, specs, and description
- trigram: `pg_trgm` title similarity within normalized brand blocks
- vector: pgvector cosine search, gated by cheap lexical/trigram evidence so embeddings are not global by default

The output is a map of product-index pairs to retrieval evidence strings such as `exact:ean:...`, `lexical:fts:0.5000`, `trigram:title:0.9000`, and `vector:cosine:0.8800`.

Trade-off: retrieval mostly decides which pairs are worth scoring. Strong exact evidence also feeds a high-precision merge policy, but only after URL trust checks and contradiction guards.

## 3. Dense Semantic Similarity

Before scoring, the pipeline embeds every product that appears in at least one candidate pair. It then computes pairwise cosine similarity for every scored candidate pair and stores that value as `semantic_sim`.

Trade-off: this avoids embedding truly unrelated products, but logistic regression still receives a dense semantic feature for all candidate pairs it scores.

## 4. Pairwise ML Features

`src/cartsy_dedupe/features.py` builds the stable pairwise feature contract ported from the experiment:

```text
same_retailer
brand_exact
brand_fuzzy
title_token_set
title_partial
category_exact
model_token_jaccard
salient_token_jaccard
size_match
size_conflict
pack_match
pack_conflict
price_ratio_diff
price_both_present
identifier_any
exact_global_id
exact_ean
exact_gtin
exact_upc
exact_asin
exact_retailer_sku
exact_canonical_url
exact_key_count
exact_evidence_strength
exact_sku_same_retailer
exact_sku_cross_retailer
rule_score
rule_auto_blocked
lexical_sim
trigram_sim
semantic_sim
retrieval_layer_count
variant_conflict
```

The logistic model is trained against this exact column order. Existing bundles can still load with their saved feature subset, but new exact-evidence columns only affect the ML score after retraining and reviewing `feature_coefficients.csv`.

## 5. Merge Decision

The merge policy is layered:

```python
features = build_pair_features(...)
ml_score = logistic_regression.predict_proba(features)

if hard_contradiction:
    decision = "no_merge"
elif strong_exact_evidence:
    decision = "merge"
elif ml_score >= threshold:
    decision = "merge"
else:
    decision = "no_merge"
```

Strong exact evidence means shared global identifiers, ASIN, same-retailer SKU, or a trusted canonical product URL. Canonical URLs are only inserted into the exact-key table when they look like product pages; click/count/redirect/tracking URLs are ignored.

Hard contradictions include deterministic size conflicts, pack conflicts, variant conflicts, the existing rule scorer's hard blocks, and any still-present structured attribute conflict. LLM attribute extraction is currently disabled, so normal runtime contradictions come from deterministic and ML feature evidence.

Trade-off: exact identifiers recover the high-precision deterministic behavior that should not be vetoed by a weakly calibrated model, while logistic regression remains the inspectable decision surface for ambiguous lexical, trigram, vector, semantic, and rule-score evidence.

## 6. Cluster Accepted Merge Edges

Accepted merge pairs become graph edges. `src/cartsy_dedupe/clustering.py` unions connected components into final `dedupe_id` groups and keeps cluster-level guards against unsafe connected-component spillover.

Trade-off: the model scores pairs, while clustering handles group construction. A cluster can be blocked even when an individual edge looks attractive if the group-level evidence becomes contradictory.

## 7. Training And Evaluation

Training should use the augmented dataset from the experiment checkout when available:

```bash
cartsy-dedupe train-model \
  --products data/dataset_v1_augmented.csv \
  --ground-truth data/ground_truth_v1_augmented.csv \
  --output-dir models \
  --target-precision 0.97 \
  --max-positive-pairs 10000 \
  --max-hard-negative-pairs 30000 \
  --use-openai-embeddings
```

Synthetic augmentation creates two high-value patterns from the experiment:

- guarded positive duplicates that preserve variant signatures
- dirty-identifier hard negatives with shared weak identifiers and variant conflicts

If the augmented CSVs need to be regenerated, `cartsy-dedupe augment-training-data` ports those same patterns into this repo.

Every training run writes threshold curves, precision/recall/F1, false positives, false negatives, feature coefficients, and top risky predicted clusters. These artifacts are the calibration surface for changing thresholds or features.

When exact-evidence features change, retrain the model before using the new columns as ML signals. The runtime exact-merge policy works independently of retraining because it reads the same feature dictionary directly.

## 8. Retrieval Defaults

The production recall profile should keep all retrieval layers enabled:

```bash
CARTSY_FTS_CANDIDATES=25
CARTSY_TRIGRAM_CANDIDATES=25
CARTSY_TRIGRAM_MIN_SIMILARITY=0.55
CARTSY_VECTOR_CANDIDATES=25
```

Trade-off: this creates more candidates than the smoke-test profile, but candidate generation had not been the observed bottleneck in the regression runs. The loss was in merge policy, so retrieval should stay recall-oriented after exact behavior is restored.

## 9. Caching Policy

Stage caching is disabled in the main run path while the ML scorer is being integrated. This prevents stale retrieval/scoring/cluster artifacts from masking model or feature changes.

Product embedding caching remains available because it is keyed by product embedding text, model, dimensions, and code fingerprint. It saves OpenAI cost without hiding scorer behavior.

## 10. Run Artifacts

Each run writes:

- `normalized_products.parquet`
- `candidate_pairs.parquet`
- `product_assignments.csv`
- `dedupe_groups.jsonl`
- `near_miss_pairs.csv`
- `summary_report.json`

The durable artifact files are the source of truth for completed-run search, explanations, and API responses.
