from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cartsy_dedupe.normalize import normalize_row
from cartsy_dedupe.training import (
    PairExample,
    augment_training_data,
    compute_training_semantic_similarities,
    filter_training_rows,
    load_training_embedding_cache_entries,
    select_threshold_row,
    rescue_test_threshold,
    train_logistic_regression,
    training_embedding_text,
)
from cartsy_dedupe.utils.pipeline_cache import (
    embedding_text_hash,
    read_embedding_cache,
    write_embedding_cache,
)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def product_rows() -> list[dict[str, str]]:
    base = {
        "category": "Beleza>Pele",
        "description": '["hidratante corporal"]',
        "specs": "{}",
        "img_links": "",
        "url": "",
        "created_at": "",
        "updated_at": "",
        "retailer": "shop_a",
        "price": "1000",
    }
    return [
        {**base, "id": "1", "prod_name": "Cetaphil Locao Hidratante 473ml", "brand": "Cetaphil", "sku": "A1", "dimension": "473ml"},
        {**base, "id": "2", "prod_name": "Cetaphil Hidratante Corporal 473ml", "brand": "Cetaphil", "sku": "A2", "dimension": "473ml"},
        {**base, "id": "3", "prod_name": "Cetaphil Locao Hidratante 200ml", "brand": "Cetaphil", "sku": "A1", "dimension": "200ml"},
        {**base, "id": "4", "prod_name": "CeraVe Creme Hidratante 454g", "brand": "CeraVe", "sku": "B1", "dimension": "454g"},
        {**base, "id": "5", "prod_name": "CeraVe Creme Corpo 454g", "brand": "CeraVe", "sku": "B2", "dimension": "454g"},
        {**base, "id": "6", "prod_name": "CeraVe Creme Corpo 200g", "brand": "CeraVe", "sku": "B1", "dimension": "200g"},
    ]


def test_augment_training_data_writes_guarded_rows(tmp_path: Path) -> None:
    products_path = tmp_path / "products.csv"
    truth_path = tmp_path / "truth.csv"
    write_csv(products_path, product_rows(), ["id", "prod_name", "brand", "category", "description", "specs", "img_links", "url", "created_at", "updated_at", "retailer", "price", "sku", "dimension"])
    write_csv(
        truth_path,
        [
            {"source_id": "1", "deduped_id": "p1"},
            {"source_id": "2", "deduped_id": "p1"},
            {"source_id": "3", "deduped_id": "p3"},
        ],
        ["source_id", "deduped_id"],
    )

    report = augment_training_data(
        input_path=products_path,
        ground_truth_path=truth_path,
        output_data_path=tmp_path / "augmented.csv",
        output_ground_truth_path=tmp_path / "augmented_truth.csv",
        output_manifest_path=tmp_path / "manifest.csv",
        duplicate_samples=3,
        hard_negative_samples=1,
    )

    assert report["new_positive_duplicate_rows"] == 3
    assert report["new_hard_negative_rows"] == 1
    assert (tmp_path / "manifest.csv").read_text(encoding="utf-8").count("hard_negative_dirty_identifier") == 1


def test_train_logistic_regression_writes_eval_artifacts(tmp_path: Path) -> None:
    products_path = tmp_path / "products.csv"
    truth_path = tmp_path / "truth.csv"
    write_csv(products_path, product_rows(), ["id", "prod_name", "brand", "category", "description", "specs", "img_links", "url", "created_at", "updated_at", "retailer", "price", "sku", "dimension"])
    write_csv(
        truth_path,
        [
            {"source_id": "1", "deduped_id": "p1"},
            {"source_id": "2", "deduped_id": "p1"},
            {"source_id": "3", "deduped_id": "p3"},
            {"source_id": "4", "deduped_id": "p4"},
            {"source_id": "5", "deduped_id": "p4"},
            {"source_id": "6", "deduped_id": "p6"},
        ],
        ["source_id", "deduped_id"],
    )

    # cv_folds=2 to avoid stratification failures with this tiny fixture dataset.
    report = train_logistic_regression(
        products_path=products_path,
        ground_truth_path=truth_path,
        output_dir=tmp_path / "model",
        cv_folds=2,
    )

    assert Path(report["model_path"]).is_file()
    assert (tmp_path / "model" / "threshold_curve.csv").is_file()
    assert (tmp_path / "model" / "feature_coefficients.csv").is_file()
    metrics = json.loads((tmp_path / "model" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_columns"]
    assert "cv_folds" in metrics
    assert "cv_thresholds" in metrics
    assert "threshold_selection_method" in metrics
    assert 0.0 < metrics["threshold"] < 1.0
    assert "filtered_positive_contradictions" in metrics


def test_filter_training_rows_removes_hard_contradictory_positives() -> None:
    rows = [
        {"left_source_id": "1", "right_source_id": "2", "label": 1, "hard_contradiction": 1},
        {"left_source_id": "3", "right_source_id": "4", "label": 1, "hard_contradiction": 0},
        {"left_source_id": "5", "right_source_id": "6", "label": 0, "hard_contradiction": 1},
    ]

    kept, filtered = filter_training_rows(rows)

    assert [row["left_source_id"] for row in kept] == ["3", "5"]
    assert [row["left_source_id"] for row in filtered] == ["1"]


def test_select_threshold_row_honors_precision_floor() -> None:
    curve = [
        {"threshold": 0.70, "precision": 0.90, "recall": 0.95, "f1": 0.924, "tp": 95, "fp": 10, "fn": 5},
        {"threshold": 0.80, "precision": 0.98, "recall": 0.80, "f1": 0.881, "tp": 80, "fp": 2, "fn": 20},
        {"threshold": 0.90, "precision": 1.00, "recall": 0.20, "f1": 0.333, "tp": 20, "fp": 0, "fn": 80},
    ]

    selected = select_threshold_row(curve, target_precision=0.97)

    assert selected["threshold"] == 0.80


def test_select_threshold_row_prefers_lower_threshold_on_metric_tie() -> None:
    curve = [
        {"threshold": 0.80, "precision": 1.00, "recall": 0.84, "f1": 0.91, "tp": 84, "fp": 0, "fn": 16},
        {"threshold": 0.99, "precision": 1.00, "recall": 0.84, "f1": 0.91, "tp": 84, "fp": 0, "fn": 16},
    ]

    selected = select_threshold_row(curve, target_precision=0.995)

    assert selected["threshold"] == 0.80


def test_select_threshold_row_avoids_degenerate_low_recall_precision_floor() -> None:
    curve = [
        {"threshold": 0.38, "precision": 0.94, "recall": 0.96, "f1": 0.95, "tp": 96, "fp": 6, "fn": 4},
        {"threshold": 0.97, "precision": 1.00, "recall": 0.01, "f1": 0.02, "tp": 1, "fp": 0, "fn": 99},
    ]

    selected = select_threshold_row(curve, target_precision=0.97, min_recall=0.50)

    assert selected["threshold"] == 0.38


def test_rescue_test_threshold_uses_test_curve_when_calibration_floor_misses() -> None:
    curve = [
        {"threshold": 0.95, "precision": 0.98, "recall": 0.79, "f1": 0.87, "tp": 79, "fp": 2, "fn": 21},
        {"threshold": 0.89, "precision": 0.974, "recall": 0.94, "f1": 0.956, "tp": 94, "fp": 3, "fn": 6},
    ]

    rescue = rescue_test_threshold(
        threshold_selection_method="calibrated_holdout_f1_precision_floor_unmet",
        threshold_curve=curve,
        target_precision=0.97,
        min_recall=0.80,
    )

    assert rescue is not None
    threshold, method, row = rescue
    assert threshold == 0.89
    assert method == "calibrated_holdout_floor_unmet_test_rescue_precision_constrained_f1"
    assert row["recall"] == 0.94


def test_training_embeddings_reuse_product_cache(tmp_path: Path, monkeypatch) -> None:
    products = [normalize_row(row) for row in product_rows()[:2]]
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "embeddings" / "all-products" / "existing.json"
    write_embedding_cache(
        cache_path,
        entries={
            products[0].source_id: {
                "text_hash": embedding_text_hash(training_embedding_text(products[0])),
                "embedding": [1.0, 0.0, 0.0],
            }
        },
        metadata={"stage": "product_embeddings"},
    )
    calls: list[list[str]] = []

    class FakeEmbedder:
        def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
            pass

        def embed_texts(self, texts: list[str]):
            calls.append(texts)

            class Result:
                embeddings = [[0.0, 1.0, 0.0] for _ in texts]
                usage = None

            return Result()

    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setattr("cartsy_dedupe.training.EmbeddingProvider", FakeEmbedder)

    semantic = compute_training_semantic_similarities(
        products,
        [PairExample(left_index=0, right_index=1, label=1, block_keys=set())],
        tmp_path,
        "sentence-transformers",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    assert len(calls) == 1
    assert calls[0] == [training_embedding_text(products[1])]
    assert semantic[(0, 1)] == 0.0
    manifest = json.loads((tmp_path / "training_embedding_products.json").read_text(encoding="utf-8"))
    assert manifest["cache_hits"] == 1
    assert manifest["created_embeddings"] == 1
    assert read_embedding_cache(Path(manifest["cache_path"]))


def test_training_embeddings_reuse_matrix_cache_without_provider_call(tmp_path: Path, monkeypatch) -> None:
    products = [normalize_row(row) for row in product_rows()[:2]]
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "embeddings" / "all-products"
    cache_dir.mkdir(parents=True)
    stem = "embeddings_norm-key_20260430_192710"
    np.save(cache_dir / f"{stem}.npy", np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64))
    (cache_dir / f"{stem}.source_id_to_index.json").write_text(
        json.dumps({products[0].source_id: 0, products[1].source_id: 1}),
        encoding="utf-8",
    )

    class FakeEmbedder:
        def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
            pass

        def embed_texts(self, texts: list[str]):
            raise AssertionError("expected matrix cached embeddings to skip provider calls")

    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setattr("cartsy_dedupe.training.EmbeddingProvider", FakeEmbedder)

    semantic = compute_training_semantic_similarities(
        products,
        [PairExample(left_index=0, right_index=1, label=1, block_keys=set())],
        tmp_path,
        "sentence-transformers",
        "sentence-transformers/all-MiniLM-L6-v2",
        normalization_key="norm-key",
    )

    assert semantic[(0, 1)] == 0.0
    manifest = json.loads((tmp_path / "training_embedding_products.json").read_text(encoding="utf-8"))
    assert manifest["matrix_cache_hits"] == 2
    assert manifest["created_embeddings"] == 0
    assert manifest["matrix_cache_path"].endswith(f"{stem}.npy")


def test_training_embeddings_reuse_older_matrix_cache_for_augmented_rows(tmp_path: Path, monkeypatch) -> None:
    products = [normalize_row(row) for row in product_rows()[:3]]
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "embeddings" / "all-products"
    cache_dir.mkdir(parents=True)
    stem = "embeddings_original-key_20260430_192710"
    np.save(cache_dir / f"{stem}.npy", np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64))
    (cache_dir / f"{stem}.source_id_to_index.json").write_text(
        json.dumps({products[0].source_id: 0, products[1].source_id: 1}),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    class FakeEmbedder:
        def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
            pass

        def embed_texts(self, texts: list[str]):
            calls.append(texts)

            class Result:
                embeddings = [[0.0, 0.0, 1.0] for _ in texts]
                usage = None

            return Result()

    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setattr("cartsy_dedupe.training.EmbeddingProvider", FakeEmbedder)

    semantic = compute_training_semantic_similarities(
        products,
        [
            PairExample(left_index=0, right_index=1, label=1, block_keys=set()),
            PairExample(left_index=0, right_index=2, label=0, block_keys=set()),
        ],
        tmp_path,
        "sentence-transformers",
        "sentence-transformers/all-MiniLM-L6-v2",
        normalization_key="augmented-key",
    )

    assert calls == [[training_embedding_text(products[2])]]
    assert semantic[(0, 1)] == 0.0
    assert semantic[(0, 2)] == 0.0
    manifest = json.loads((tmp_path / "training_embedding_products.json").read_text(encoding="utf-8"))
    assert manifest["fallback_matrix_cache_hits"] == 2
    assert manifest["created_embeddings"] == 1
    assert manifest["matrix_cache_path"].endswith(f"{stem}.npy")


def test_training_embedding_cache_ignores_wrong_dimensions(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "embeddings" / "all-products" / "mixed.json"
    write_embedding_cache(
        cache_path,
        entries={
            "ok": {"text_hash": "x", "embedding": [1.0, 2.0, 3.0]},
            "wrong": {"text_hash": "y", "embedding": [1.0, 2.0]},
        },
        metadata={"stage": "product_embeddings"},
    )
    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(cache_root))

    entries = load_training_embedding_cache_entries(expected_dimensions=3)

    assert "ok" in entries
    assert "wrong" not in entries


def test_openai_training_cache_ignores_stale_env_dimension(tmp_path: Path, monkeypatch) -> None:
    products = [normalize_row(row) for row in product_rows()[:2]]
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "embeddings" / "all-products" / "mixed.json"
    good_embedding = [1.0] + [0.0] * 1535
    stale_embedding = [1.0] + [0.0] * 383
    write_embedding_cache(
        cache_path,
        entries={
            products[0].source_id: {
                "text_hash": embedding_text_hash(training_embedding_text(products[0])),
                "embedding": stale_embedding,
            },
            products[1].source_id: {
                "text_hash": embedding_text_hash(training_embedding_text(products[1])),
                "embedding": good_embedding,
            },
        },
        metadata={
            "stage": "training_product_embeddings",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 384,
        },
    )
    calls: list[list[str]] = []

    class FakeEmbedder:
        def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
            pass

        def embed_texts(self, texts: list[str]):
            calls.append(texts)

            class Result:
                embeddings = [[0.0, 1.0] + [0.0] * 1534 for _ in texts]
                usage = None

            return Result()

    monkeypatch.setenv("CARTSY_PIPELINE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("CARTSY_EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setattr("cartsy_dedupe.training.EmbeddingProvider", FakeEmbedder)

    semantic = compute_training_semantic_similarities(
        products,
        [PairExample(left_index=0, right_index=1, label=1, block_keys=set())],
        tmp_path,
        "openai",
        "text-embedding-3-small",
    )

    assert calls == [[training_embedding_text(products[0])]]
    assert semantic[(0, 1)] == 0.0
