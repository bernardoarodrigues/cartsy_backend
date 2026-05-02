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

`src/cartsy_dedupe/features.py` builds the stable pairwise feature contract. `DEFAULT_FEATURE_COLUMNS` is the model contract: adding, removing, or reordering columns invalidates existing `.joblib` bundles and requires retraining.

| Feature | Meaning |
|---|---|
| `same_retailer` | 1 if both products come from the same retailer |
| `brand_exact` | 1 if normalized brand strings are identical |
| `brand_fuzzy` | Levenshtein ratio of normalized brand strings |
| `title_token_set` | Token-set ratio of normalized product titles |
| `title_partial` | Partial ratio of normalized product titles |
| `category_exact` | 1 if leaf category segments are identical |
| `model_token_jaccard` | Jaccard similarity of extracted alphanumeric model tokens |
| `salient_token_jaccard` | Jaccard similarity of title tokens after removing brand, category, stopwords, and digit tokens |
| `size_match` | 1 if both products have an unambiguous size and they are equivalent |
| `size_conflict` | 1 if both products have an unambiguous size and they differ |
| `pack_match` | 1 if both products have an explicit pack count and they agree |
| `pack_conflict` | 1 if both products have an explicit pack count and they differ |
| `price_ratio_diff` | Absolute relative price difference — `|p1-p2| / max(p1,p2)` |
| `price_both_present` | 1 if both products have a non-null price |
| `identifier_any` | 1 if any shared identifier was found (in-product or from retrieval evidence) |
| `exact_global_id` | 1 if a shared EAN, GTIN, or UPC was found in retrieval evidence |
| `exact_ean` | 1 if product-level EAN values agree |
| `exact_gtin` | 1 if product-level GTIN values agree |
| `exact_upc` | 1 if product-level UPC values agree |
| `exact_asin` | 1 if ASIN values agree (product-level or retrieval evidence) |
| `exact_retailer_sku` | 1 if a same-retailer SKU key was found in retrieval evidence |
| `exact_canonical_url` | 1 if a canonical product URL key was found in retrieval evidence |
| `exact_key_count` | Number of distinct exact identifier types matched |
| `exact_evidence_strength` | Scalar strength of strongest exact evidence (1.0 for global ID, 0.92 for ASIN, …) |
| `exact_sku_same_retailer` | 1 if SKU matches within the same retailer |
| `exact_sku_cross_retailer` | 1 if SKU matches across different retailers |
| `rule_certain_match` | 1 if `CERTAIN_MATCH` fired (EAN/GTIN/UPC/ASIN/URL) |
| `rule_strong_match` | 1 if `STRONG_MATCH` fired (retailer SKU or brand+title≥0.95 with model overlap) |
| `rule_likely_match` | 1 if `LIKELY_MATCH` fired (brand+title≥0.85+size, or brand+model overlap+title≥0.70) |
| `rule_certain_block` | 1 if `CERTAIN_BLOCK` fired (hard contradiction detected) |
| `lexical_sim` | Normalized FTS rank from the lexical retrieval layer |
| `trigram_sim` | Trigram title similarity from the trigram retrieval layer |
| `semantic_sim` | Cosine similarity of dense product embeddings |
| `retrieval_layer_count` | Number of distinct retrieval layers (exact/lexical/trigram/vector) that surfaced the pair |
| `variant_conflict` | 1 if same brand but salient title tokens are disjoint (likely a color/shade variant) |
| `feature_coverage_count` | Count of indicator features carrying non-zero signal; low values flag sparse-evidence pairs |

## 5. Rule Evaluation And Merge Decision

Each candidate pair first passes through an ordered condition chain in `src/cartsy_dedupe/scoring.py`. The chain returns one of five certainty levels:

| Level | What it means | Pipeline action |
|---|---|---|
| `CERTAIN_BLOCK` | Hard contradiction (conflicting global ID, brand, size, or pack count) | score=0.0, skip ML |
| `CERTAIN_MATCH` | Exact global ID (EAN/GTIN/UPC), ASIN, or trusted canonical URL | score=1.0, skip ML |
| `STRONG_MATCH` | Same-retailer SKU, or brand+title≥0.95 with model overlap | ML scores pair; certainty is a feature |
| `LIKELY_MATCH` | Brand+title≥0.85+size match, or brand+model overlap+title≥0.70 | ML scores pair; certainty is a feature |
| `UNCERTAIN` | No clear signal either way | ML scores pair; certainty is a feature |

For pairs that reach the ML model:

```python
rule_decision = evaluate_rule(left, right)
pair_features = build_pair_features(..., rule_decision=rule_decision)
ml_score = calibrated_logistic_regression.predict_proba(pair_features)

if rule_decision.certainty == CERTAIN_MATCH:
    decision = "merge"                          # bypass ML
elif rule_decision.certainty == CERTAIN_BLOCK:
    decision = "no_merge"                       # bypass ML
elif hard_contradiction_features(pair_features):
    decision = "no_merge"                       # ML called, but score capped
elif ml_score >= threshold:
    decision = "merge"
else:
    decision = "no_merge"
```

Canonical URLs are trusted only when they look like product pages; click/count/redirect/tracking paths are filtered by `canonicalize_url` before insertion into the exact-key table.

Trade-off: deterministic certainty conditions handle the obvious cases without ML inference overhead. The calibrated logistic regression remains the decision surface for everything in between, with interpretable rule indicator features rather than a single blended score float.

## 6. Cluster Accepted Merge Edges

Accepted merge pairs become graph edges. `src/cartsy_dedupe/clustering.py` unions connected components into final `dedupe_id` groups and keeps cluster-level guards against unsafe connected-component spillover.

Trade-off: the model scores pairs, while clustering handles group construction. A cluster can be blocked even when an individual edge looks attractive if the group-level evidence becomes contradictory.

## 7. Training And Evaluation

Training should use the augmented dataset:

```bash
cartsy-dedupe train-model \
  --products data/dataset_v1_augmented.csv \
  --ground-truth data/ground_truth_v1_augmented.csv \
  --output-dir models \
  --cv-folds 5 \
  --max-positive-pairs 10000 \
  --max-hard-negative-pairs 30000 \
  --use-embeddings
```

`--target-precision` is accepted for backward compatibility but no longer drives threshold selection.

**Threshold selection** uses stratified k-fold cross-validation (default 5 folds). The median F1-maximising threshold across folds is chosen, making the decision boundary stable across data distributions rather than fitted to a single test fold. A separate held-out calibration split (≈15% of data) is used for `CalibratedClassifierCV` isotonic regression, which makes `P(merge)` values reliable probabilities rather than raw logit-derived scores on imbalanced data. For small datasets the function automatically falls back to a simpler 70/30 split.

Synthetic augmentation creates two high-value patterns:

- guarded positive duplicates that preserve variant signatures
- dirty-identifier hard negatives with shared weak identifiers and variant conflicts

If the augmented CSVs need to be regenerated, `cartsy-dedupe augment-training-data` ports those same patterns.

Every training run writes threshold curves, precision/recall/F1, CV thresholds, false positives, false negatives, feature coefficients, and top risky predicted clusters. `metrics.json` includes `cv_folds`, `cv_thresholds`, and `calibrated` keys.

When `DEFAULT_FEATURE_COLUMNS` changes, retrain the model — the runtime `load_ml_model` check validates that bundle feature columns match the current contract and rejects stale bundles.

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

Full stage-level caching (reading/writing complete retrieval, scoring, and cluster results) is not implemented in the main run path. Each run recomputes all stages from scratch.

Per-layer retrieval caching is available inside `load_or_fetch_retrieval_rows`. Pass `retrieval_env` and `retrieval_code` to `generate_candidate_pairs` to enable it; by default these are `None` and the layer cache is skipped.

Product embedding caching is active by default. It is keyed by product embedding text, model, dimensions, and code fingerprint, so embeddings are reused across repeated runs on the same products and save OpenAI API cost.

## 10. Run Artifacts

Each run writes:

- `normalized_products.parquet`
- `candidate_pairs.parquet`
- `product_assignments.csv`
- `dedupe_groups.jsonl`
- `near_miss_pairs.csv`
- `summary_report.json`

The durable artifact files are the source of truth for completed-run search, explanations, and API responses.
