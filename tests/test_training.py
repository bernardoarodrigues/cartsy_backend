from __future__ import annotations

import csv
import json
from pathlib import Path

from cartsy_dedupe.training import augment_training_data, train_logistic_regression


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

    report = train_logistic_regression(products_path=products_path, ground_truth_path=truth_path, output_dir=tmp_path / "model")

    assert Path(report["model_path"]).is_file()
    assert (tmp_path / "model" / "threshold_curve.csv").is_file()
    assert (tmp_path / "model" / "feature_coefficients.csv").is_file()
    metrics = json.loads((tmp_path / "model" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_columns"]
