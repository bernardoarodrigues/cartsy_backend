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

from cartsy_dedupe.embeddings import EmbeddingProvider, configured_embedding_model, embedding_provider_name
from cartsy_dedupe.features import DEFAULT_FEATURE_COLUMNS, build_pair_features, feature_vector, hard_contradiction_features
from cartsy_dedupe.ingest import load_rows
from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.schemas import NormalizedProduct
from cartsy_dedupe.scoring import evaluate_rule, string_similarity
from cartsy_dedupe.utils.pipeline_helpers import embedding_text
from cartsy_dedupe.utils.pipeline_sql import postgres_retrieval_features

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
SHADE_WORDS = ["Preto", "Branco", "Azul", "Rosa", "Prata", "Dourado", "Nude", "Claro", "Escuro", "Natural"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairExample:
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
) -> dict[str, object]:
    """Train, calibrate, and evaluate a logistic-regression pair scorer.

    Splits data into train (≈70%), calibration (≈15%), and test (≈15%) sets.
    Threshold selection uses ``cv_folds``-fold stratified cross-validation on
    the training set, picking the median F1-maximising threshold across folds.
    After threshold selection the base model is calibrated on the held-out
    calibration split using ``CalibratedClassifierCV(method='isotonic')``,
    making ``P(merge)`` values reliable so that the threshold is meaningful
    rather than artificially pushed toward extremes.

    For small datasets where the three-split or CV would fail, the function
    falls back to a simpler 70/30 split with a single F1-optimal threshold.

    Parameters
    ----------
    target_precision:
        Retained for backward compatibility — stored in the metrics JSON but
        no longer used to select the decision threshold.
    cv_folds:
        Number of stratified CV folds for threshold selection.  Automatically
        clamped down when the training set is too small.
    """
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    from joblib import dump
    from sklearn.calibration import CalibratedClassifierCV
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
        )
    else:
        logger.info("Skipping embeddings (lexical and structural features only)")
        semantic_by_pair = {}
    logger.info("Computing pair feature rows (%d pairs)", len(pair_examples))
    rows = pair_feature_rows(products, pair_examples, semantic_by_pair)
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
    if train_pos >= effective_folds and train_neg >= effective_folds:
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        for fold_train_local, fold_val_local in cv.split(x_train_scaled, y[train_idx]):
            fold_model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state)
            fold_model.fit(x_train_scaled[fold_train_local], y[train_idx][fold_train_local])
            fold_scores = fold_model.predict_proba(x_train_scaled[fold_val_local])[:, 1]
            curve = build_threshold_curve(y[train_idx][fold_val_local], fold_scores)
            best = max(curve, key=lambda r: r["f1"])
            cv_thresholds.append(float(best["threshold"]))
        threshold = float(np.median(cv_thresholds))
        logger.info("CV threshold selection (%d folds): thresholds=%s median=%.4f", effective_folds, cv_thresholds, threshold)
    else:
        # Too few samples for reliable CV — use F1-optimal on test set as fallback.
        logger.warning("Skipping CV threshold (too few samples per class); using F1-optimal on test set")
        fallback_scores = base_model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
        fallback_curve = build_threshold_curve(y[test_idx], fallback_scores)
        threshold = float(max(fallback_curve, key=lambda r: r["f1"])["threshold"])
        cv_thresholds = [threshold]

    # ── Probability calibration ────────────────────────────────────────────────
    # Calibrating with isotonic regression makes P(merge) values reliable so
    # that the threshold corresponds to an actual probability rather than a raw
    # logit-derived score that may be pushed to extremes on imbalanced data.
    can_calibrate = len(cal_idx) >= 4 and int((y[cal_idx] == 1).sum()) >= 1 and int((y[cal_idx] == 0).sum()) >= 1
    if can_calibrate:
        logger.info("Calibrating model on %d held-out calibration samples", len(cal_idx))
        model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
        model.fit(scaler.transform(x[cal_idx]), y[cal_idx])
    else:
        logger.info("Skipping calibration (calibration set too small or absent)")
        model = base_model  # type: ignore[assignment]

    # ── Evaluation on test set ─────────────────────────────────────────────────
    logger.info("Evaluating on test set (%d pairs, threshold=%.4f)", len(test_idx), threshold)
    test_scores = model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
    threshold_curve = build_threshold_curve(y[test_idx], test_scores)
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
    write_feature_coefficients(output_path / "feature_coefficients.csv", base_model.coef_[0], DEFAULT_FEATURE_COLUMNS)
    write_error_examples(output_path / "false_positives.csv", rows, test_idx, y[test_idx], test_scores, test_pred, want_label=0, want_pred=1)
    write_error_examples(output_path / "false_negatives.csv", rows, test_idx, y[test_idx], test_scores, test_pred, want_label=1, want_pred=0)
    write_risky_clusters(output_path / "top_risky_clusters.csv", rows, test_idx, test_scores, test_pred)
    report = {
        "model_path": str(model_path),
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "threshold": threshold,
        "target_precision": target_precision,
        "cv_folds": effective_folds,
        "cv_thresholds": cv_thresholds,
        "calibrated": can_calibrate,
        "train_pairs": int(len(train_idx)),
        "calibration_pairs": int(len(cal_idx)),
        "test_pairs": int(len(test_idx)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int(len(y) - y.sum()),
        "test_average_precision": float(average_precision),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "use_embeddings": use_embeddings,
        "embedding_provider": resolved_embedding_provider,
        "embedding_model": resolved_embedding_model,
        "artifacts": [
            "threshold_curve.csv",
            "feature_coefficients.csv",
            "false_positives.csv",
            "false_negatives.csv",
            "top_risky_clusters.csv",
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
    pair_key = (min(left_index, right_index), max(left_index, right_index))
    if pair_key in pairs:
        return False
    inferred_label = int(labels_by_index[pair_key[0]] == labels_by_index[pair_key[1]]) if label is None else label
    pairs[pair_key] = PairExample(pair_key[0], pair_key[1], inferred_label, infer_block_keys(products[pair_key[0]], products[pair_key[1]]))
    return True


def infer_block_keys(left: NormalizedProduct, right: NormalizedProduct) -> set[str]:
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


def compute_training_semantic_similarities(
    products: list[NormalizedProduct],
    pair_examples: list[PairExample],
    output_path: Path,
    embedding_provider: str,
    embedding_model: str,
) -> dict[tuple[int, int], float]:
    embedder = EmbeddingProvider(provider=embedding_provider, model=embedding_model)
    indexes = sorted({index for pair in pair_examples for index in (pair.left_index, pair.right_index)})
    texts = [
        embedding_text(
            brand=products[index].brand_raw,
            title=products[index].name_raw,
            category=products[index].category_raw,
            description=products[index].description_raw,
            specs=products[index].specs_raw,
            dimension=products[index].dimension_raw,
        )
        for index in indexes
    ]
    embeddings: dict[int, list[float]] = {}
    batch_offsets = range(0, len(indexes), 128)
    if len(indexes) > 128:
        batch_offsets = tqdm(batch_offsets, desc=f"{embedding_provider} embedding batches", unit="batch")
    for offset in batch_offsets:
        batch_indexes = indexes[offset : offset + 128]
        result = embedder.embed_texts(texts[offset : offset + 128])
        for index, embedding in zip(batch_indexes, result.embeddings, strict=True):
            embeddings[index] = embedding
    (output_path / "training_embedding_products.json").write_text(
        json.dumps(
            {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "source_ids": [products[index].source_id for index in indexes],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    pair_iter: Any = pair_examples
    if len(pair_examples) > 500:
        pair_iter = tqdm(pair_examples, desc="Pair cosine similarity", unit="pair")
    semantic: dict[tuple[int, int], float] = {}
    for pair in pair_iter:
        if pair.left_index in embeddings and pair.right_index in embeddings:
            semantic[(pair.left_index, pair.right_index)] = cosine(
                embeddings[pair.left_index], embeddings[pair.right_index]
            )
    return semantic


def build_threshold_curve(y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
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


def choose_threshold(curve: list[dict[str, float]], target_precision: float) -> float:
    eligible = [row for row in curve if row["precision"] >= target_precision and row["tp"] > 0]
    if eligible:
        return float(max(eligible, key=lambda row: (row["recall"], row["f1"]))["threshold"])
    return float(max(curve, key=lambda row: row["f1"])["threshold"])


def read_truth(path: str | Path) -> dict[str, str]:
    return {row["source_id"]: row["deduped_id"] for row in load_rows(path)}


def write_threshold_curve(path: Path, rows: list[dict[str, float]]) -> None:
    write_csv(path, rows, ["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])


def write_feature_coefficients(path: Path, coefficients: np.ndarray, columns: list[str]) -> None:
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
    out: list[dict[str, Any]] = []
    for local_pos, row_index in enumerate(test_idx):
        if int(y_true[local_pos]) == want_label and int(pred[local_pos]) == want_pred:
            row = dict(rows[int(row_index)])
            row["score"] = float(scores[local_pos])
            out.append(row)
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    columns = ["left_source_id", "right_source_id", "label", "score", "hard_contradiction", "blocking_keys", *DEFAULT_FEATURE_COLUMNS]
    write_csv(path, out[:500], columns)


def write_risky_clusters(path: Path, rows: list[dict[str, Any]], test_idx: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> None:
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
                    "size_conflict": row["size_conflict"],
                    "pack_conflict": row["pack_conflict"],
                    "blocking_keys": row["blocking_keys"],
                }
            )
    risky.sort(key=lambda item: float(item["score"]), reverse=True)
    write_csv(
        path,
        risky[:200],
        ["left_source_id", "right_source_id", "score", "label", "hard_contradiction", "variant_conflict", "size_conflict", "pack_conflict", "blocking_keys"],
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def weighted_base_pool(rows: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    weighted: list[dict[str, str]] = []
    for row in rows:
        weight = 1 + int(bool(row.get("sku"))) + int(bool(row.get("description"))) + int(bool(row.get("dimension")))
        weighted.extend([row] * weight)
    rng.shuffle(weighted)
    return weighted or rows


def copy_product_row(row: dict[str, str], source_id: int) -> dict[str, str]:
    copied = {column: row.get(column, "") for column in PRODUCT_COLUMNS}
    copied["id"] = str(source_id)
    return copied


def next_available_id(start: int, existing: set[str]) -> int:
    value = start
    while str(value) in existing:
        value += 1
    return value


def mutate_description_light(row: dict[str, str], rng: random.Random) -> str:
    text = row.get("description", "")
    if not text:
        row["description"] = '["Produto original com detalhes resumidos pelo varejista."]'
    else:
        row["description"] = text.replace("COMPRE AGORA", "Disponivel para compra").replace("Descubra", "Conheca")
    return "description_light_rewrite"


def mutate_missing_field(row: dict[str, str], rng: random.Random) -> str:
    candidates = [field for field in ["brand", "description", "specs", "dimension", "img_links", "sku", "price"] if row.get(field)]
    if not candidates:
        return "missing:none"
    field = rng.choice(candidates)
    row[field] = ""
    return f"missing:{field}"


def mutate_title_order(row: dict[str, str], rng: random.Random) -> str:
    pieces = [piece.strip() for piece in re.split(r"[,|-]", row.get("prod_name", "")) if piece.strip()]
    if len(pieces) >= 2:
        rng.shuffle(pieces)
        row["prod_name"] = " - ".join(pieces)
    else:
        row["prod_name"] = re.sub(r"\s+", " ", row.get("prod_name", "").replace(" com ", " c/ ")).strip()
    return "title_order_variation"


def mutate_price_and_retailer(row: dict[str, str], rng: random.Random) -> str:
    try:
        cents = int(float(str(row.get("price", "")).replace(",", ".")))
        row["price"] = str(max(1, int(round(cents * (1 + rng.uniform(-0.12, 0.12))))))
    except ValueError:
        pass
    if row.get("retailer") and rng.random() < 0.35:
        row["retailer"] = row["retailer"].replace("_", "-")
    return "same_identifier_price_jitter"


def mutate_identifier_presence(row: dict[str, str], rng: random.Random) -> str:
    if row.get("sku"):
        row["sku"] = "" if rng.random() < 0.6 else row["sku"]
    return "identifier_missing_or_specs_sparse"


def mutate_brand_case(row: dict[str, str], rng: random.Random) -> str:
    if row.get("brand"):
        row["brand"] = row["brand"].upper() if rng.random() < 0.5 else row["brand"].title()
    return "brand_case_and_spacing"


def variant_signature(row: dict[str, str]) -> dict[str, object]:
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
    for key, before_value in before.items():
        if before_value is not None and after.get(key) != before_value:
            return False
    return True


def make_weak_shared_sku(row: dict[str, str]) -> None:
    if not row.get("sku"):
        row["sku"] = re.sub(r"\W+", "", row.get("brand", "").upper())[:8] + "-SHARED"


def apply_variant_conflict(row: dict[str, str], rng: random.Random) -> str:
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
    shade = rng.choice(SHADE_WORDS)
    row["prod_name"] = f"{title} Cor {shade}".strip()
    return f"shade:{shade}"


def cosine(left: list[float], right: list[float]) -> float:
    left_array = np.array(left, dtype=float)
    right_array = np.array(right, dtype=float)
    denom = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left_array, right_array) / denom)))
