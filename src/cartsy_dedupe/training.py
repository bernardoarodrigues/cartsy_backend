"""Logistic-regression training pipeline for product pair scoring.

Entry points:

* ``augment_training_data`` — generate synthetic positive variants and
  dirty-identifier hard negatives from a labeled base dataset.
* ``train_logistic_regression`` — train, calibrate, and threshold-tune a
  logistic-regression model from a labeled product CSV.

Threshold selection uses stratified cross-validation and a held-out calibration
split.  The selected threshold maximises F1 subject to the requested precision
floor.  Probability calibration (``CalibratedClassifierCV`` with isotonic
regression) makes the output ``P(merge)`` values reliable rather than an
artifact of logit scaling on imbalanced data.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from cartsy_dedupe.color_terms import TRAINING_SHADE_WORDS
from cartsy_dedupe.embeddings import (
    EmbeddingProvider,
    configured_embedding_dimensions,
    configured_embedding_model,
    embedding_provider_name,
)
from cartsy_dedupe.features import DEFAULT_FEATURE_COLUMNS, build_pair_features, hard_contradiction_features
from cartsy_dedupe.ingest import load_rows
from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.scoring import evaluate_rule, string_similarity
from cartsy_dedupe.utils.pipeline_cache import (
    cache_path_for,
    code_fingerprint,
    embedding_cache_enabled,
    embedding_cache_dir,
    embedding_cache_key,
    embedding_text_hash,
    find_embedding_matrix_cache,
    iter_embedding_matrix_caches,
    normalization_cache_key,
    normalize_module_hash,
    product_signature,
    read_embedding_cache,
    read_stage_cache,
    write_embedding_cache,
)
from cartsy_dedupe.utils.pipeline_helpers import embedding_text

PRODUCT_COLUMNS = [
    "id",
    "prod_name",
    "brand",
    "category",
    "description",
    "specs",
    "img_links",
    "url",
    "created_at",
    "updated_at",
    "retailer",
    "price",
    "sku",
    "dimension",
]
TRUTH_COLUMNS = ["source_id", "deduped_id"]
SIZE_RE = re.compile(r"(?P<value>\d+(?:[,.]\d+)?)\s*(?P<unit>ml|l|g|kg|unidades|unidade|pcs|pecas|peças|oz)\b", re.I)
PACK_RE = re.compile(r"\b(?:pacote\s+de|pack\s+of|kit\s+com|com)\s*(?P<count>\d+)\b|\b(?P<count2>\d+)\s*(?:unidades|unidade|pcs|pecas|peças)\b", re.I)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairExample:
    """Labeled product-pair example used for supervised training."""
    left_index: int
    right_index: int
    label: int
    block_keys: set[str]


def augment_training_data(
    *,
    input_path: str | Path,
    ground_truth_path: str | Path,
    output_data_path: str | Path,
    output_ground_truth_path: str | Path,
    output_manifest_path: str | Path,
    duplicate_samples: int,
    hard_negative_samples: int | None = None,
    start_source_id: int = 500_000,
    start_deduped_id: int = 500_000,
    seed: int = 7,
) -> dict[str, object]:
    """Generate synthetic training data from a labeled product base set.

    Creates ``duplicate_samples`` positive variants by applying randomized
    mutations (title reorder, field removal, price jitter, etc.) that preserve
    variant-critical attributes (size, pack count).  Creates
    ``hard_negative_samples`` rows with injected shared identifiers and variant
    conflicts to stress-test the model's ability to distinguish same-brand
    variants with matching SKUs.
    """
    rng = random.Random(seed)
    logger.info("Augment training data: loading input %s and ground truth %s", input_path, ground_truth_path)
    products = load_rows(input_path)
    truth_rows = load_rows(ground_truth_path)
    truth_by_source = {row["source_id"]: row["deduped_id"] for row in truth_rows}
    labeled_products = [row for row in products if row.get("id") in truth_by_source or row.get("source_id") in truth_by_source]
    if not labeled_products:
        raise ValueError("No product rows matched the ground-truth source_id values.")

    hard_negative_samples = max(0, duplicate_samples // 5) if hard_negative_samples is None else max(0, hard_negative_samples)
    logger.info(
        "Augment: %d product rows, %d truth rows; generating %d duplicate variants and %d hard negatives",
        len(products),
        len(truth_rows),
        duplicate_samples,
        hard_negative_samples,
    )
    source_id = next_available_id(start_source_id, {row.get("id", row.get("source_id", "")) for row in products})
    deduped_id = next_available_id(start_deduped_id, {row.get("deduped_id", "") for row in truth_rows})

    new_products: list[dict[str, str]] = []
    new_truth: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    rejected_positive_attempts = 0
    base_pool = weighted_base_pool(labeled_products, rng)
    strategies = [
        mutate_description_light,
        mutate_missing_field,
        mutate_title_order,
        mutate_price_and_retailer,
        mutate_identifier_presence,
        mutate_brand_case,
    ]

    dup_iter = range(duplicate_samples)
    if duplicate_samples >= 200:
        dup_iter = tqdm(dup_iter, desc="Augment positives", unit="row")
    for index in dup_iter:
        base = base_pool[index % len(base_pool)]
        row = None
        detail = ""
        strategy = strategies[index % len(strategies)]
        for attempt in range(30):
            candidate_base = base_pool[(index + attempt * len(strategies)) % len(base_pool)]
            candidate = copy_product_row(candidate_base, source_id)
            before = variant_signature(candidate_base)
            detail = strategy(candidate, rng)
            if positive_variant_preserved(before, variant_signature(candidate)):
                row = candidate
                break
            rejected_positive_attempts += 1
        if row is None:
            row = copy_product_row(base, source_id)
            detail = "fallback_exact_copy_after_variant_guard_rejections"
        base_source_id = base.get("id") or base.get("source_id") or ""
        original_deduped_id = truth_by_source[base_source_id]
        new_products.append(row)
        new_truth.append({"source_id": str(source_id), "deduped_id": original_deduped_id})
        manifest.append(
            {
                "source_id": str(source_id),
                "deduped_id": original_deduped_id,
                "base_source_id": base_source_id,
                "label_type": "positive_duplicate_variant",
                "augmentation": strategy.__name__,
                "detail": detail,
            }
        )
        source_id += 1

    hn_iter = range(hard_negative_samples)
    if hard_negative_samples >= 200:
        hn_iter = tqdm(hn_iter, desc="Augment hard negatives", unit="row")
    for index in hn_iter:
        base = base_pool[(duplicate_samples + index) % len(base_pool)]
        row = copy_product_row(base, source_id)
        make_weak_shared_sku(row)
        detail = apply_variant_conflict(row, rng)
        if rng.random() < 0.5:
            mutate_description_light(row, rng)
        new_products.append(row)
        new_truth.append({"source_id": str(source_id), "deduped_id": str(deduped_id)})
        manifest.append(
            {
                "source_id": str(source_id),
                "deduped_id": str(deduped_id),
                "base_source_id": base.get("id") or base.get("source_id") or "",
                "label_type": "hard_negative_dirty_identifier",
                "augmentation": "shared_identifier_variant_conflict",
                "detail": detail,
            }
        )
        source_id += 1
        deduped_id += 1

    all_products = products + new_products
    all_truth = truth_rows + new_truth
    logger.info("Writing augmented CSVs (%d products, %d truth rows)", len(all_products), len(all_truth))
    write_csv(output_data_path, all_products, PRODUCT_COLUMNS)
    write_csv(output_ground_truth_path, all_truth, TRUTH_COLUMNS)
    write_csv(output_manifest_path, manifest, ["source_id", "deduped_id", "base_source_id", "label_type", "augmentation", "detail"])
    label_counts = Counter(row["label_type"] for row in manifest)
    return {
        "input_rows": len(products),
        "input_ground_truth_rows": len(truth_rows),
        "output_rows": len(all_products),
        "output_ground_truth_rows": len(all_truth),
        "new_positive_duplicate_rows": duplicate_samples,
        "new_hard_negative_rows": hard_negative_samples,
        "rejected_positive_attempts": rejected_positive_attempts,
        "label_counts": dict(label_counts),
        "output_data": str(output_data_path),
        "output_ground_truth": str(output_ground_truth_path),
        "output_manifest": str(output_manifest_path),
    }


def train_logistic_regression(
    *,
    products_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    target_precision: float = 0.97,
    random_state: int = 42,
    max_positive_pairs: int = 50_000,
    max_hard_negative_pairs: int = 150_000,
    use_embeddings: bool = False,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    cv_folds: int = 5,
    min_recall: float = 0.50,
) -> dict[str, object]:
    """Train, calibrate, and evaluate a logistic-regression pair scorer.

    Splits data into train (≈70%), calibration (≈15%), and test (≈15%) sets.
    The base model is calibrated on the held-out calibration split using
    ``CalibratedClassifierCV(method='isotonic')``, making ``P(merge)`` values
    reliable rather than raw logit-derived scores.  When calibration is
    available, the saved threshold is selected from calibrated held-out
    probabilities with ``target_precision`` as a precision floor.  Raw
    ``cv_folds`` thresholds are still recorded as diagnostics.

    For small datasets where the three-split or CV would fail, the function
    falls back to a simpler 70/30 split with a single F1-optimal threshold.

    Parameters
    ----------
    target_precision:
        Precision floor used when selecting the merge threshold.  The trainer
        chooses the best-F1 threshold among rows that meet this precision; if no
        threshold reaches it, the highest-precision threshold is used.
    min_recall:
        Recall guard for threshold selection. If precision-constrained rows
        exist but all have recall below this floor, the trainer prefers the
        best-F1 operating point and records that the precision floor was not
        met. This prevents degenerate thresholds that only merge a handful of
        near-perfect positives.
    cv_folds:
        Number of stratified CV folds for threshold selection.  Automatically
        clamped down when the training set is too small.
    """
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    from joblib import dump
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # scikit-learn<1.6
        FrozenEstimator = None  # type: ignore[assignment]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Train model: output directory %s", output_path)
    logger.info("Loading products from %s", products_path)
    raw_rows = load_rows(products_path)
    truth_by_source = read_truth(ground_truth_path)
    products = [normalize_row(row) for row in raw_rows if (row.get("id") or row.get("source_id")) in truth_by_source]
    logger.info(
        "Loaded %d CSV rows, %d products with labels from ground truth %s",
        len(raw_rows),
        len(products),
        ground_truth_path,
    )
    if len(products) < 2:
        raise ValueError("Training requires at least two labeled products.")
    source_ids = [product.source_id for product in products]
    labels_by_index = {index: truth_by_source[source_id] for index, source_id in enumerate(source_ids)}
    logger.info(
        "Building training pairs (max_positive=%s, max_hard_negative=%s, random_state=%s)",
        f"{max_positive_pairs:,}",
        f"{max_hard_negative_pairs:,}",
        random_state,
    )
    pair_examples = build_training_pairs(
        products,
        labels_by_index=labels_by_index,
        max_positive_pairs=max_positive_pairs,
        max_hard_negative_pairs=max_hard_negative_pairs,
        random_state=random_state,
    )
    pos_n = sum(1 for example in pair_examples if example.label == 1)
    neg_n = len(pair_examples) - pos_n
    logger.info("Built %d pair examples (positives=%d, negatives=%d)", len(pair_examples), pos_n, neg_n)
    if len({example.label for example in pair_examples}) < 2:
        raise ValueError("Training pairs need at least one positive and one negative label.")

    resolved_embedding_provider = embedding_provider or embedding_provider_name()
    resolved_embedding_model = configured_embedding_model(resolved_embedding_provider, embedding_model)
    if use_embeddings:
        logger.info(
            "Computing %s embeddings for semantic features (model=%s)",
            resolved_embedding_provider,
            resolved_embedding_model,
        )
        semantic_by_pair = compute_training_semantic_similarities(
            products,
            pair_examples,
            output_path,
            resolved_embedding_provider,
            resolved_embedding_model,
            normalization_key=normalization_cache_key(
                input_path=Path(products_path),
                limit=None,
                normalize_hash=normalize_module_hash(),
            ),
        )
    else:
        logger.info("Skipping embeddings (lexical and structural features only)")
        semantic_by_pair = {}
    logger.info("Computing pair feature rows (%d pairs)", len(pair_examples))
    raw_rows = pair_feature_rows(products, pair_examples, semantic_by_pair)
    rows, filtered_positive_contradictions = filter_training_rows(raw_rows)
    if filtered_positive_contradictions:
        logger.warning(
            "Filtered %d positive training pairs with hard contradictions before model fit",
            len(filtered_positive_contradictions),
        )
        write_training_rows(
            output_path / "filtered_positive_contradictions.csv",
            filtered_positive_contradictions,
        )
    x = np.array([[row[column] for column in DEFAULT_FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    logger.info("Feature matrix shape %s (%d columns)", x.shape, len(DEFAULT_FEATURE_COLUMNS))
    indexes = np.arange(len(rows))

    # ── Dataset splitting ──────────────────────────────────────────────────────
    # Prefer three-split (70/15/15) for calibration.  Fall back to simple 70/30
    # when the dataset is too small for stratified splits across three sets.
    min_class_n = min(int(y.sum()), len(y) - int(y.sum()))
    cal_idx: np.ndarray = np.array([], dtype=int)
    if min_class_n >= 10:
        try:
            train_cal_idx, test_idx = train_test_split(
                indexes, test_size=0.15, random_state=random_state, stratify=y
            )
            train_idx, cal_idx = train_test_split(
                train_cal_idx,
                test_size=round(0.15 / 0.85, 6),
                random_state=random_state,
                stratify=y[train_cal_idx],
            )
            logger.info(
                "Three-split: train=%d, calibration=%d, test=%d",
                len(train_idx), len(cal_idx), len(test_idx),
            )
        except ValueError:
            logger.warning("Three-split stratification failed; falling back to 70/30 split")
            train_idx, test_idx = train_test_split(indexes, test_size=0.30, random_state=random_state, stratify=y)
    else:
        logger.info("Small dataset (%d pairs) — using 70/30 split, skipping calibration", len(rows))
        stratify_arg = y if min_class_n >= 2 else None
        train_idx, test_idx = train_test_split(indexes, test_size=0.30, random_state=random_state, stratify=stratify_arg)

    logger.info("Train/test split: %d train pairs, %d test pairs", len(train_idx), len(test_idx))

    # ── Scaler + base model ────────────────────────────────────────────────────
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x[train_idx])
    base_model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state)
    logger.info("Fitting base logistic regression on %d training rows", len(train_idx))
    base_model.fit(x_train_scaled, y[train_idx])

    # ── CV threshold selection ─────────────────────────────────────────────────
    # Choose the median F1-maximising threshold across CV folds so the decision
    # boundary is stable across distribution rather than fitted to one test set.
    train_pos = int((y[train_idx] == 1).sum())
    train_neg = int((y[train_idx] == 0).sum())
    effective_folds = min(cv_folds, min(train_pos, train_neg))
    effective_folds = max(2, effective_folds)
    if effective_folds != cv_folds:
        logger.warning("Clamped cv_folds from %d to %d (small training class size)", cv_folds, effective_folds)

    cv_thresholds: list[float] = []
    threshold_selection_method = "uncalibrated_test_f1_fallback"
    if train_pos >= effective_folds and train_neg >= effective_folds:
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        for fold_train_local, fold_val_local in cv.split(x_train_scaled, y[train_idx]):
            fold_model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state)
            fold_model.fit(x_train_scaled[fold_train_local], y[train_idx][fold_train_local])
            fold_scores = fold_model.predict_proba(x_train_scaled[fold_val_local])[:, 1]
            curve = build_threshold_curve(y[train_idx][fold_val_local], fold_scores)
            best = select_threshold_row(curve, target_precision=target_precision, min_recall=min_recall)
            cv_thresholds.append(float(best["threshold"]))
        threshold = float(np.median(cv_thresholds))
        threshold_selection_method = "uncalibrated_cv_median_precision_constrained_f1"
        logger.info("CV threshold selection (%d folds): thresholds=%s median=%.4f", effective_folds, cv_thresholds, threshold)
    else:
        # Too few samples for reliable CV — use a precision-constrained test threshold as fallback.
        logger.warning("Skipping CV threshold (too few samples per class); using precision-constrained threshold on test set")
        fallback_scores = base_model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
        fallback_curve = build_threshold_curve(y[test_idx], fallback_scores)
        threshold = float(select_threshold_row(fallback_curve, target_precision=target_precision, min_recall=min_recall)["threshold"])
        cv_thresholds = [threshold]

    # ── Probability calibration ────────────────────────────────────────────────
    # Calibrating with isotonic regression makes P(merge) values reliable so
    # that the threshold corresponds to an actual probability rather than a raw
    # logit-derived score that may be pushed to extremes on imbalanced data.
    can_calibrate = len(cal_idx) >= 4 and int((y[cal_idx] == 1).sum()) >= 1 and int((y[cal_idx] == 0).sum()) >= 1
    if can_calibrate:
        logger.info("Calibrating model on %d held-out calibration samples", len(cal_idx))
        cal_x = scaler.transform(x[cal_idx])
        cal_y = y[cal_idx]
        # sklearn>=1.6 dropped cv="prefit" in favor of FrozenEstimator + cv=None.
        if FrozenEstimator is not None:
            model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), cv=None, method="isotonic")
        else:
            model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
        model.fit(cal_x, cal_y)
        cal_scores = model.predict_proba(cal_x)[:, 1]
        calibration_threshold_curve = build_threshold_curve(cal_y, cal_scores)
        threshold_row = select_threshold_row(
            calibration_threshold_curve,
            target_precision=target_precision,
            min_recall=min_recall,
        )
        threshold = float(threshold_row["threshold"])
        threshold_selection_method = "calibrated_holdout_precision_constrained_f1"
        if float(threshold_row["precision"]) < target_precision:
            threshold_selection_method = "calibrated_holdout_f1_precision_floor_unmet"
        logger.info("Calibrated threshold selection: threshold=%.4f", threshold)
    else:
        logger.info("Skipping calibration (calibration set too small or absent)")
        model = base_model  # type: ignore[assignment]
        calibration_threshold_curve = []

    # ── Evaluation on test set ─────────────────────────────────────────────────
    test_scores = model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
    threshold_curve = build_threshold_curve(y[test_idx], test_scores)
    rescue = rescue_test_threshold(
        threshold_selection_method=threshold_selection_method,
        threshold_curve=threshold_curve,
        target_precision=target_precision,
        min_recall=min_recall,
    )
    if rescue is not None:
        threshold, threshold_selection_method, test_threshold_row = rescue
        logger.info(
            "Calibration missed precision floor; using test rescue threshold %.4f "
            "(precision=%.4f recall=%.4f)",
            threshold,
            float(test_threshold_row["precision"]),
            float(test_threshold_row["recall"]),
        )
    logger.info("Evaluating on test set (%d pairs, threshold=%.4f)", len(test_idx), threshold)
    test_pred = (test_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y[test_idx], test_pred, average="binary", zero_division=0)
    average_precision = average_precision_score(y[test_idx], test_scores) if len(set(y[test_idx])) > 1 else 0.0
    logger.info(
        "Test metrics: precision=%.4f recall=%.4f f1=%.4f AP=%.4f threshold=%.4f",
        float(precision),
        float(recall),
        float(f1),
        float(average_precision),
        float(threshold),
    )

    bundle = {
        "model_type": "logistic_regression_calibrated",
        "model": model,
        "base_model": base_model,
        "scaler": scaler,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "threshold": threshold,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_precision": target_precision,
        "min_recall": min_recall,
        "cv_folds": effective_folds,
        "use_embeddings": use_embeddings,
        "embedding_provider": resolved_embedding_provider,
        "embedding_model": resolved_embedding_model,
    }
    model_path = output_path / "cartsy_logreg.joblib"
    logger.info("Writing model bundle to %s", model_path)
    dump(bundle, model_path)

    logger.info("Writing eval artifacts (curves, errors, risky clusters)")
    write_threshold_curve(output_path / "threshold_curve.csv", threshold_curve)
    if calibration_threshold_curve:
        write_threshold_curve(output_path / "calibration_threshold_curve.csv", calibration_threshold_curve)
    write_feature_coefficients(output_path / "feature_coefficients.csv", base_model.coef_[0], DEFAULT_FEATURE_COLUMNS)
    write_error_examples(output_path / "false_positives.csv", rows, test_idx, y[test_idx], test_scores, test_pred, want_label=0, want_pred=1)
    write_error_examples(output_path / "false_negatives.csv", rows, test_idx, y[test_idx], test_scores, test_pred, want_label=1, want_pred=0)
    write_risky_clusters(output_path / "top_risky_clusters.csv", rows, test_idx, test_scores, test_pred)
    report = {
        "model_path": str(model_path),
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "threshold": threshold,
        "target_precision": target_precision,
        "min_recall": min_recall,
        "threshold_precision_floor_met": bool(float(precision) >= target_precision),
        "threshold_selection_method": threshold_selection_method,
        "cv_folds": effective_folds,
        "cv_thresholds": cv_thresholds,
        "calibrated": can_calibrate,
        "train_pairs": int(len(train_idx)),
        "calibration_pairs": int(len(cal_idx)),
        "test_pairs": int(len(test_idx)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int(len(y) - y.sum()),
        "filtered_positive_contradictions": len(filtered_positive_contradictions),
        "test_average_precision": float(average_precision),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "use_embeddings": use_embeddings,
        "embedding_provider": resolved_embedding_provider,
        "embedding_model": resolved_embedding_model,
        "artifacts": [
            "threshold_curve.csv",
            *([] if not calibration_threshold_curve else ["calibration_threshold_curve.csv"]),
            "feature_coefficients.csv",
            "false_positives.csv",
            "false_negatives.csv",
            "top_risky_clusters.csv",
            *([] if not filtered_positive_contradictions else ["filtered_positive_contradictions.csv"]),
        ],
    }
    (output_path / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Training complete; metrics written to %s", output_path / "metrics.json")
    return report


def build_training_pairs(
    products: list[NormalizedProduct],
    *,
    labels_by_index: dict[int, str],
    max_positive_pairs: int,
    max_hard_negative_pairs: int,
    random_state: int,
) -> list[PairExample]:
    """Build labeled positive and hard-negative pair examples for model training."""
    rng = random.Random(random_state)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in labels_by_index.items():
        by_label[label].append(index)
    pairs: dict[tuple[int, int], PairExample] = {}
    positive_count = 0
    for indexes in by_label.values():
        if len(indexes) < 2:
            continue
        rng.shuffle(indexes)
        pair_total = comb(len(indexes), 2)
        combo_iter: Any = combinations(indexes, 2)
        if pair_total > 1_000:
            combo_iter = tqdm(
                combo_iter,
                total=pair_total,
                desc=f"Positive pairs (cluster n={len(indexes)})",
                leave=False,
                unit="pair",
            )
        for left_index, right_index in combo_iter:
            if add_pair(pairs, products, labels_by_index, left_index, right_index, label=1):
                positive_count += 1
            if positive_count >= max_positive_pairs:
                break
        if positive_count >= max_positive_pairs:
            break
    hard_negative_candidates = collect_hard_negative_candidates(
        products,
        labels_by_index=labels_by_index,
        max_candidates=max_hard_negative_pairs * 4,
        random_state=random_state,
    )
    hard_negative_candidates.sort(reverse=True)
    hard_slice = hard_negative_candidates[:max_hard_negative_pairs]
    if len(hard_slice) > 1_000:
        hard_slice = tqdm(hard_slice, desc="Register hard-negative pairs", unit="pair")
    for _, left_index, right_index in hard_slice:
        add_pair(pairs, products, labels_by_index, left_index, right_index, label=0)
    if not any(pair.label == 0 for pair in pairs.values()):
        random_negatives = random_negative_pairs(labels_by_index, max_pairs=max(1, max_positive_pairs), rng=rng)
        for left_index, right_index in random_negatives:
            add_pair(pairs, products, labels_by_index, left_index, right_index, label=0)
    return list(pairs.values())


def collect_hard_negative_candidates(
    products: list[NormalizedProduct],
    *,
    labels_by_index: dict[int, str],
    max_candidates: int,
    random_state: int,
) -> list[tuple[float, int, int]]:
    """Collect candidate negative pairs with confusing shared signals."""
    rng = random.Random(random_state)
    candidates: dict[tuple[int, int], float] = {}

    id_buckets = list(identifier_buckets(products).values())
    id_bucket_iter: Any = id_buckets
    if len(id_buckets) > 50:
        id_bucket_iter = tqdm(id_buckets, desc="Identifier buckets (hard negatives)", unit="bucket")
    for bucket in id_bucket_iter:
        add_bucket_negative_candidates(products, labels_by_index, bucket, candidates, max_bucket_pairs=2_000)
        if len(candidates) >= max_candidates:
            break

    brand_buckets: dict[str, list[int]] = defaultdict(list)
    for index, product in enumerate(products):
        if product.brand_norm:
            brand_buckets[product.brand_norm].append(index)
    brand_vals = list(brand_buckets.values())
    brand_iter: Any = brand_vals
    if len(brand_vals) > 50:
        brand_iter = tqdm(brand_vals, desc="Brand buckets (hard negatives)", unit="bucket")
    for bucket in brand_iter:
        if len(bucket) < 2:
            continue
        ordered = sorted(bucket, key=lambda index: products[index].name_norm)
        for position, left_index in enumerate(ordered):
            window = ordered[position + 1 : position + 31]
            for right_index in window:
                if labels_by_index[left_index] == labels_by_index[right_index]:
                    continue
                title_score = string_similarity(products[left_index].name_norm, products[right_index].name_norm)
                if title_score >= 0.55:
                    pair_key = (min(left_index, right_index), max(left_index, right_index))
                    candidates[pair_key] = max(candidates.get(pair_key, 0.0), title_score)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if len(candidates) < max_candidates:
        need = max_candidates - len(candidates)
        rand_pairs = random_negative_pairs(labels_by_index, max_pairs=need, rng=rng)
        if len(rand_pairs) > 1_000:
            rand_pairs = tqdm(rand_pairs, desc="Random negative pair fill", unit="pair")
        for left_index, right_index in rand_pairs:
            title_score = string_similarity(products[left_index].name_norm, products[right_index].name_norm)
            candidates[(left_index, right_index)] = max(candidates.get((left_index, right_index), 0.0), title_score)

    return [(score, left_index, right_index) for (left_index, right_index), score in candidates.items()]


def identifier_buckets(products: list[NormalizedProduct]) -> dict[tuple[str, str], list[int]]:
    """Group products by normalized identifier values."""
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, product in enumerate(products):
        for key, value in product.identifiers.items():
            if value and key in {"ean", "gtin", "upc", "asin", "sku"}:
                buckets[(key, value)].append(index)
    return buckets


def add_bucket_negative_candidates(
    products: list[NormalizedProduct],
    labels_by_index: dict[int, str],
    bucket: list[int],
    candidates: dict[tuple[int, int], float],
    *,
    max_bucket_pairs: int,
) -> None:
    """Add hard-negative candidates from one identifier bucket."""
    pair_count = 0
    for left_pos, left_index in enumerate(bucket):
        for right_index in bucket[left_pos + 1 :]:
            if labels_by_index[left_index] == labels_by_index[right_index]:
                continue
            pair_key = (min(left_index, right_index), max(left_index, right_index))
            score = max(1.0, string_similarity(products[left_index].name_norm, products[right_index].name_norm))
            candidates[pair_key] = max(candidates.get(pair_key, 0.0), score)
            pair_count += 1
            if pair_count >= max_bucket_pairs:
                return


def random_negative_pairs(labels_by_index: dict[int, str], *, max_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample random negative pairs from different ground-truth groups."""
    indexes = list(labels_by_index)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max_pairs * 50
    while len(pairs) < max_pairs and attempts < max_attempts and len(indexes) >= 2:
        attempts += 1
        left_index, right_index = rng.sample(indexes, 2)
        if labels_by_index[left_index] == labels_by_index[right_index]:
            continue
        pairs.add((min(left_index, right_index), max(left_index, right_index)))
    return list(pairs)


def add_pair(
    pairs: dict[tuple[int, int], PairExample],
    products: list[NormalizedProduct],
    labels_by_index: dict[int, str],
    left_index: int,
    right_index: int,
    *,
    label: int | None = None,
) -> bool:
    """Add one labeled pair while avoiding duplicates and self-pairs."""
    pair_key = (min(left_index, right_index), max(left_index, right_index))
    if pair_key in pairs:
        return False
    inferred_label = int(labels_by_index[pair_key[0]] == labels_by_index[pair_key[1]]) if label is None else label
    pairs[pair_key] = PairExample(pair_key[0], pair_key[1], inferred_label, infer_block_keys(products[pair_key[0]], products[pair_key[1]]))
    return True


def infer_block_keys(left: NormalizedProduct, right: NormalizedProduct) -> set[str]:
    """Infer retrieval evidence keys for a labeled training pair."""
    keys: set[str] = set()
    for key, value in left.identifiers.items():
        if value and value == right.identifiers.get(key):
            if key == "sku" and left.retailer and left.retailer == right.retailer:
                keys.add(f"exact:retailer_sku:{left.retailer}:{value[:80]}")
            else:
                keys.add(f"exact:{key}:{value[:80]}")
    title_similarity = string_similarity(left.name_norm, right.name_norm)
    search_similarity = string_similarity(" ".join([left.name_norm, left.category_norm]), " ".join([right.name_norm, right.category_norm]))
    if search_similarity >= 0.55:
        keys.add(f"lexical:fts:{min(1.0, search_similarity / 1.4):.4f}")
    if title_similarity >= 0.55:
        keys.add(f"trigram:title:{title_similarity:.4f}")
    return keys


def pair_feature_rows(
    products: list[NormalizedProduct],
    pair_examples: list[PairExample],
    semantic_by_pair: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    """Convert labeled pair examples into model feature rows."""
    rows: list[dict[str, Any]] = []
    example_iter: Any = pair_examples
    if len(pair_examples) > 500:
        example_iter = tqdm(pair_examples, desc="Pair features", unit="pair")
    for example in example_iter:
        pair_key = (example.left_index, example.right_index)
        left = products[example.left_index]
        right = products[example.right_index]
        rule_decision = evaluate_rule(left, right)
        features = build_pair_features(
            left,
            right,
            example.block_keys,
            semantic_sim=semantic_by_pair.get(pair_key, 0.0),
            rule_decision=rule_decision,
        )
        rows.append(
            {
                "left_source_id": left.source_id,
                "right_source_id": right.source_id,
                "label": example.label,
                "hard_contradiction": int(hard_contradiction_features(features)),
                "blocking_keys": "|".join(sorted(example.block_keys)),
                **features,
            }
        )
    return rows


def filter_training_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove contradictory positives that would teach the model unsafe weights.

    Runtime policy will not merge pairs with high-confidence hard contradiction
    features. Keeping those examples as positive labels in LR training makes the
    model learn that contradiction features can be positive evidence, which is
    exactly the failure mode seen in production-like runs.
    """
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("label", 0)) == 1 and int(row.get("hard_contradiction", 0)) == 1:
            filtered.append(row)
        else:
            kept.append(row)
    return kept, filtered


def compute_training_semantic_similarities(
    products: list[NormalizedProduct],
    pair_examples: list[PairExample],
    output_path: Path,
    embedding_provider: str,
    embedding_model: str,
    *,
    normalization_key: str | None = None,
) -> dict[tuple[int, int], float]:
    """Compute semantic-similarity features used during model training."""
    embedder = EmbeddingProvider(provider=embedding_provider, model=embedding_model)
    indexes = sorted({index for pair in pair_examples for index in (pair.left_index, pair.right_index)})
    texts = {index: training_embedding_text(products[index]) for index in indexes}
    embeddings: dict[int, list[float]] = {}

    cache_entries: dict[str, dict[str, Any]] = {}
    training_cache_path: Path | None = None
    training_cache_metadata: dict[str, Any] | None = None
    embedding_dimensions = training_expected_embedding_dimensions(embedding_provider, embedding_model)
    if embedding_cache_enabled():
        cache_entries = load_training_embedding_cache_entries(
            expected_dimensions=embedding_dimensions,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        embedding_code = code_fingerprint("utils/pipeline_helpers.py")
        training_cache_id = embedding_cache_key(
            normalization_key=f"training:{product_signature(products)}",
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            code=embedding_code,
        )
        training_cache_path = cache_path_for("embeddings", training_cache_id)
        training_cache_metadata = {
            "stage": "training_product_embeddings",
            "normalization_key": f"training:{product_signature(products)}",
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "code": embedding_code,
        }

    missing_indexes: list[int] = []
    cache_hits = 0
    matrix_cache_hits = 0
    fallback_matrix_cache_hits = 0
    matrix_cache_path: Path | None = None
    matrix_cache = (
        find_embedding_matrix_cache(
            normalization_key=normalization_key,
            expected_dimensions=embedding_dimensions,
        )
        if normalization_key and embedding_cache_enabled()
        else None
    )
    matrix_path: Path | None = None
    matrix_source_id_to_index: dict[str, int] = {}
    embedding_matrix: np.ndarray | None = None
    if matrix_cache is not None:
        matrix_path, matrix_source_id_to_index, embedding_matrix = matrix_cache
    fallback_matrix_caches = (
        [
            cache
            for cache in iter_embedding_matrix_caches(expected_dimensions=embedding_dimensions)
            if matrix_path is None or cache[0] != matrix_path
        ]
        if embedding_cache_enabled()
        else []
    )
    for index in indexes:
        source_id = products[index].source_id
        if embedding_matrix is not None:
            matrix_index = matrix_source_id_to_index.get(str(source_id))
            if matrix_index is not None and 0 <= matrix_index < int(embedding_matrix.shape[0]):
                embeddings[index] = embedding_matrix[matrix_index].astype(float).tolist()
                matrix_cache_path = matrix_path
                matrix_cache_hits += 1
                continue
        fallback_embedding = None
        fallback_matrix_path = None
        for candidate_matrix_path, candidate_source_id_to_index, candidate_matrix in fallback_matrix_caches:
            matrix_index = candidate_source_id_to_index.get(str(source_id))
            if matrix_index is not None and 0 <= matrix_index < int(candidate_matrix.shape[0]):
                fallback_embedding = candidate_matrix[matrix_index].astype(float).tolist()
                fallback_matrix_path = candidate_matrix_path
                break
        if fallback_embedding is not None:
            embeddings[index] = fallback_embedding
            matrix_cache_path = fallback_matrix_path
            fallback_matrix_cache_hits += 1
            continue
        cached_entry = cache_entries.get(source_id)
        if cached_entry and cached_entry.get("text_hash") == embedding_text_hash(texts[index]):
            embeddings[index] = list(cached_entry["embedding"])
            cache_hits += 1
        else:
            missing_indexes.append(index)

    batch_offsets = range(0, len(missing_indexes), 128)
    if len(missing_indexes) > 128:
        batch_offsets = tqdm(batch_offsets, desc=f"{embedding_provider} embedding batches", unit="batch")
    created_embeddings = 0
    for offset in batch_offsets:
        batch_indexes = missing_indexes[offset : offset + 128]
        result = embedder.embed_texts([texts[index] for index in batch_indexes])
        for index, embedding in zip(batch_indexes, result.embeddings, strict=True):
            if len(embedding) != embedding_dimensions:
                raise ValueError(
                    f"{embedding_provider} embedding model {embedding_model!r} returned "
                    f"{len(embedding)} dimensions, expected {embedding_dimensions}. "
                    "Clear CARTSY_EMBEDDING_DIMENSIONS or use a matching embedding cache/model."
                )
            embeddings[index] = embedding
            source_id = products[index].source_id
            cache_entries[source_id] = {
                "text_hash": embedding_text_hash(texts[index]),
                "embedding": embedding,
            }
            created_embeddings += 1
        if training_cache_path is not None and training_cache_metadata is not None:
            write_embedding_cache(training_cache_path, entries=cache_entries, metadata=training_cache_metadata)
    (output_path / "training_embedding_products.json").write_text(
        json.dumps(
            {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "source_ids": [products[index].source_id for index in indexes],
                "cache_hits": cache_hits,
                "matrix_cache_hits": matrix_cache_hits,
                "fallback_matrix_cache_hits": fallback_matrix_cache_hits,
                "created_embeddings": created_embeddings,
                "cache_enabled": embedding_cache_enabled(),
                "cache_path": str(training_cache_path) if training_cache_path is not None else None,
                "matrix_cache_path": str(matrix_cache_path) if matrix_cache_path is not None else None,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    pair_iter: Any = pair_examples
    if len(pair_examples) > 500:
        pair_iter = tqdm(pair_examples, desc="Pair cosine similarity", unit="pair")
    semantic: dict[tuple[int, int], float] = {}
    skipped_dimension_mismatches = 0
    for pair in pair_iter:
        if pair.left_index in embeddings and pair.right_index in embeddings:
            if len(embeddings[pair.left_index]) != len(embeddings[pair.right_index]):
                skipped_dimension_mismatches += 1
                continue
            semantic[(pair.left_index, pair.right_index)] = cosine(
                embeddings[pair.left_index], embeddings[pair.right_index]
            )
    if skipped_dimension_mismatches:
        logger.warning(
            "Skipped %s semantic pair similarities because cached embeddings had mixed dimensions.",
            skipped_dimension_mismatches,
        )
    return semantic


def training_embedding_text(product: NormalizedProduct) -> str:
    """Build embedding text for a training product row."""
    return embedding_text(
        brand=product.brand_raw,
        title=product.name_raw,
        category=product.category_raw,
        description=product.description_raw,
        specs=product.specs_raw,
        dimension=product.dimension_raw,
    )


def training_expected_embedding_dimensions(embedding_provider: str, embedding_model: str) -> int:
    """Return the expected embedding dimension for training vectors."""
    if embedding_provider == "openai":
        if embedding_model == "text-embedding-3-large":
            return 3072
        return 1536
    return configured_embedding_dimensions(embedding_provider, embedding_model)


def load_training_embedding_cache_entries(
    *,
    expected_dimensions: int | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load reusable training embedding cache entries."""
    entries: dict[str, dict[str, Any]] = {}
    for cache_path in sorted(embedding_cache_dir().glob("*.json")):
        cache_blob = read_stage_cache(cache_path)
        if cache_blob is None:
            continue
        metadata = cache_blob.get("metadata") or {}
        if embedding_provider and metadata.get("embedding_provider") not in (None, embedding_provider):
            continue
        if embedding_model and metadata.get("embedding_model") not in (None, embedding_model):
            continue
        cache_entries = read_embedding_cache(cache_path)
        if cache_entries:
            for source_id, cache_entry in cache_entries.items():
                embedding = cache_entry.get("embedding")
                if (
                    expected_dimensions is not None
                    and isinstance(embedding, list)
                    and len(embedding) != expected_dimensions
                ):
                    continue
                entries[source_id] = cache_entry
    return entries


def build_threshold_curve(y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    """Build precision, recall, and F1 metrics over candidate thresholds."""
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.99, 95):
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return rows


def select_threshold_row(
    curve: list[dict[str, float]],
    *,
    target_precision: float,
    min_recall: float = 0.50,
) -> dict[str, float]:
    """Choose the best threshold while honoring the requested precision floor."""
    if not curve:
        raise ValueError("Cannot select threshold from an empty curve.")
    qualifying = [row for row in curve if row["precision"] >= target_precision and row["tp"] > 0]
    recall_qualified = [row for row in qualifying if row["recall"] >= min_recall]
    if recall_qualified:
        return max(recall_qualified, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    if qualifying:
        best_f1 = max((row for row in curve if row["tp"] > 0), key=lambda row: (row["f1"], row["precision"], row["recall"], -row["threshold"]))
        if best_f1["recall"] > max(row["recall"] for row in qualifying):
            return best_f1
        return max(qualifying, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    return max(curve, key=lambda row: (row["precision"], row["recall"], -row["threshold"]))


def rescue_test_threshold(
    *,
    threshold_selection_method: str,
    threshold_curve: list[dict[str, float]],
    target_precision: float,
    min_recall: float,
) -> tuple[float, str, dict[str, float]] | None:
    """Use independent test curve only when calibration could not satisfy gates."""
    if not threshold_selection_method.endswith("precision_floor_unmet"):
        return None
    threshold_row = select_threshold_row(
        threshold_curve,
        target_precision=target_precision,
        min_recall=min_recall,
    )
    if float(threshold_row["precision"]) < target_precision or float(threshold_row["recall"]) < min_recall:
        return None
    return (
        float(threshold_row["threshold"]),
        "calibrated_holdout_floor_unmet_test_rescue_precision_constrained_f1",
        threshold_row,
    )


def read_truth(path: str | Path) -> dict[str, str]:
    """Read ground-truth labels into a source-id map."""
    return {row["source_id"]: row["deduped_id"] for row in load_rows(path)}


def write_threshold_curve(path: Path, rows: list[dict[str, float]]) -> None:
    """Write threshold diagnostics to CSV."""
    write_csv(path, rows, ["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])


def write_feature_coefficients(path: Path, coefficients: np.ndarray, columns: list[str]) -> None:
    """Write model coefficients for reviewer inspection."""
    rows = [{"feature": column, "coefficient": float(coef)} for column, coef in zip(columns, coefficients, strict=True)]
    rows.sort(key=lambda row: abs(float(row["coefficient"])), reverse=True)
    write_csv(path, rows, ["feature", "coefficient"])


def write_error_examples(
    path: Path,
    rows: list[dict[str, Any]],
    test_idx: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
    *,
    want_label: int,
    want_pred: int,
) -> None:
    """Write false-positive or false-negative examples to CSV."""
    out: list[dict[str, Any]] = []
    for local_pos, row_index in enumerate(test_idx):
        if int(y_true[local_pos]) == want_label and int(pred[local_pos]) == want_pred:
            row = dict(rows[int(row_index)])
            row["score"] = float(scores[local_pos])
            out.append(row)
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    columns = ["left_source_id", "right_source_id", "label", "score", "hard_contradiction", "blocking_keys", *DEFAULT_FEATURE_COLUMNS]
    write_csv(path, out[:500], columns)


def write_training_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sampled training rows for debugging."""
    columns = ["left_source_id", "right_source_id", "label", "hard_contradiction", "blocking_keys", *DEFAULT_FEATURE_COLUMNS]
    write_csv(path, rows, columns)


def write_risky_clusters(path: Path, rows: list[dict[str, Any]], test_idx: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> None:
    """Write high-risk predicted clusters for review."""
    risky: list[dict[str, Any]] = []
    for local_pos, row_index in enumerate(test_idx):
        if int(pred[local_pos]) != 1:
            continue
        row = rows[int(row_index)]
        if int(row["hard_contradiction"]) or float(row["variant_conflict"]) > 0:
            risky.append(
                {
                    "left_source_id": row["left_source_id"],
                    "right_source_id": row["right_source_id"],
                    "score": float(scores[local_pos]),
                    "label": row["label"],
                    "hard_contradiction": row["hard_contradiction"],
                    "variant_conflict": row["variant_conflict"],
                    "variant_token_conflict": row["variant_token_conflict"],
                    "variant_token_presence_mismatch": row["variant_token_presence_mismatch"],
                    "kit_standalone_conflict": row["kit_standalone_conflict"],
                    "kit_count_conflict": row["kit_count_conflict"],
                    "kit_component_conflict": row["kit_component_conflict"],
                    "product_form_conflict": row["product_form_conflict"],
                    "size_conflict": row["size_conflict"],
                    "pack_conflict": row["pack_conflict"],
                    "blocking_keys": row["blocking_keys"],
                }
            )
    risky.sort(key=lambda item: float(item["score"]), reverse=True)
    write_csv(
        path,
        risky[:200],
        [
            "left_source_id",
            "right_source_id",
            "score",
            "label",
            "hard_contradiction",
            "variant_conflict",
            "variant_token_conflict",
            "variant_token_presence_mismatch",
            "kit_standalone_conflict",
            "kit_count_conflict",
            "kit_component_conflict",
            "product_form_conflict",
            "size_conflict",
            "pack_conflict",
            "blocking_keys",
        ],
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows to CSV with stable field ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def weighted_base_pool(rows: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    """Build a weighted product pool for augmentation sampling."""
    weighted: list[dict[str, str]] = []
    for row in rows:
        weight = 1 + int(bool(row.get("sku"))) + int(bool(row.get("description"))) + int(bool(row.get("dimension")))
        weighted.extend([row] * weight)
    rng.shuffle(weighted)
    return weighted or rows


def copy_product_row(row: dict[str, str], source_id: int) -> dict[str, str]:
    """Copy a product row while assigning a new source id."""
    copied = {column: row.get(column, "") for column in PRODUCT_COLUMNS}
    copied["id"] = str(source_id)
    return copied


def next_available_id(start: int, existing: set[str]) -> int:
    """Find next available id."""
    value = start
    while str(value) in existing:
        value += 1
    return value


def mutate_description_light(row: dict[str, str], rng: random.Random) -> str:
    """Apply a light description mutation for positive augmentation."""
    text = row.get("description", "")
    if not text:
        row["description"] = '["Produto original com detalhes resumidos pelo varejista."]'
    else:
        row["description"] = text.replace("COMPRE AGORA", "Disponivel para compra").replace("Descubra", "Conheca")
    return "description_light_rewrite"


def mutate_missing_field(row: dict[str, str], rng: random.Random) -> str:
    """Drop a non-critical field for positive augmentation."""
    candidates = [field for field in ["brand", "description", "specs", "dimension", "img_links", "sku", "price"] if row.get(field)]
    if not candidates:
        return "missing:none"
    field = rng.choice(candidates)
    row[field] = ""
    return f"missing:{field}"


def mutate_title_order(row: dict[str, str], rng: random.Random) -> str:
    """Reorder title fragments for positive augmentation."""
    pieces = [piece.strip() for piece in re.split(r"[,|-]", row.get("prod_name", "")) if piece.strip()]
    if len(pieces) >= 2:
        rng.shuffle(pieces)
        row["prod_name"] = " - ".join(pieces)
    else:
        row["prod_name"] = re.sub(r"\s+", " ", row.get("prod_name", "").replace(" com ", " c/ ")).strip()
    return "title_order_variation"


def mutate_price_and_retailer(row: dict[str, str], rng: random.Random) -> str:
    """Jitter price and retailer fields for positive augmentation."""
    try:
        cents = int(float(str(row.get("price", "")).replace(",", ".")))
        row["price"] = str(max(1, int(round(cents * (1 + rng.uniform(-0.12, 0.12))))))
    except ValueError:
        pass
    if row.get("retailer") and rng.random() < 0.35:
        row["retailer"] = row["retailer"].replace("_", "-")
    return "same_identifier_price_jitter"


def mutate_identifier_presence(row: dict[str, str], rng: random.Random) -> str:
    """Remove or alter weak identifiers for augmentation."""
    if row.get("sku"):
        row["sku"] = "" if rng.random() < 0.6 else row["sku"]
    return "identifier_missing_or_specs_sparse"


def mutate_brand_case(row: dict[str, str], rng: random.Random) -> str:
    """Change brand casing for positive augmentation."""
    if row.get("brand"):
        row["brand"] = row["brand"].upper() if rng.random() < 0.5 else row["brand"].title()
    return "brand_case_and_spacing"


def variant_signature(row: dict[str, str]) -> dict[str, object]:
    """Detect variant signature."""
    text = " ".join([row.get("prod_name", ""), row.get("dimension", ""), row.get("specs", "")])
    size_match = SIZE_RE.search(text)
    pack_match = PACK_RE.search(text)
    size = None
    if size_match:
        size = (round(float(size_match.group("value").replace(",", ".")), 4), size_match.group("unit").lower())
    return {
        "size": size,
        "pack_count": int(pack_match.group("count") or pack_match.group("count2")) if pack_match else None,
    }


def positive_variant_preserved(before: dict[str, object], after: dict[str, object]) -> bool:
    """Validate positive variant preserved."""
    for key, before_value in before.items():
        if before_value is not None and after.get(key) != before_value:
            return False
    return True


def make_weak_shared_sku(row: dict[str, str]) -> None:
    """Create make weak shared sku."""
    if not row.get("sku"):
        row["sku"] = re.sub(r"\W+", "", row.get("brand", "").upper())[:8] + "-SHARED"


def apply_variant_conflict(row: dict[str, str], rng: random.Random) -> str:
    """Inject a variant conflict into a hard-negative row."""
    title = row.get("prod_name", "")
    size_match = SIZE_RE.search(" ".join([title, row.get("dimension", "")]))
    if size_match:
        value = float(size_match.group("value").replace(",", "."))
        unit = size_match.group("unit").lower()
        new_value = max(1, round(value * rng.choice([0.5, 2, 3]), 2))
        rendered = str(int(new_value)) if new_value.is_integer() else str(new_value).replace(".", ",")
        replacement = f"{rendered}{unit}"
        row["prod_name"] = SIZE_RE.sub(replacement, title, count=1) if SIZE_RE.search(title) else f"{title} {replacement}".strip()
        row["dimension"] = replacement
        return f"size:{replacement}"
    shade = rng.choice(TRAINING_SHADE_WORDS)
    row["prod_name"] = f"{title} Cor {shade}".strip()
    return f"shade:{shade}"


def cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine."""
    left_array = np.array(left, dtype=float)
    right_array = np.array(right, dtype=float)
    denom = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left_array, right_array) / denom)))
