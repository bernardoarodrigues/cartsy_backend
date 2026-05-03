from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from cartsy_dedupe.api import create_app
from tests.test_query import make_run_dir


def make_api_root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    root.mkdir()
    run_dir = make_run_dir(root)
    run_dir.rename(root / "run_20260430_150405")
    summary = {
        "run_id": "run_20260430_150405",
        "input_records": 2,
        "final_unique_products": 1,
        "duplicate_records_grouped": 1,
        "elapsed_seconds": 1.25,
        "metrics": {"openai": {"total_estimated_cost_usd": 0.01}},
    }
    (root / "run_20260430_150405" / "summary_report.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def test_api_lists_runs_and_summary(tmp_path: Path) -> None:
    client = TestClient(create_app(runs_root=make_api_root(tmp_path)))

    runs = client.get("/runs")
    assert runs.status_code == 200
    body = runs.json()
    assert body["runs"][0]["run_id"] == "run_20260430_150405"
    assert body["runs"][0]["model_id"] is None

    summary = client.get("/runs/run_20260430_150405/summary")
    assert summary.status_code == 200
    assert summary.json()["final_unique_products"] == 1


def test_api_filters_products_and_searches(tmp_path: Path) -> None:
    client = TestClient(create_app(runs_root=make_api_root(tmp_path)))

    products = client.get("/runs/run_20260430_150405/products", params={"brand": "cetaphil", "min_confidence": 0.9})
    assert products.status_code == 200
    assert products.json()["total"] == 2

    search = client.get("/runs/run_20260430_150405/search", params={"q": "cetaphil hidratante", "backend": "artifacts"})
    assert search.status_code == 200
    assert search.json()["results"][0]["dedupe_id"] == "prod_abc"


def test_api_group_and_explain_endpoints(tmp_path: Path) -> None:
    client = TestClient(create_app(runs_root=make_api_root(tmp_path)))

    group = client.get("/runs/run_20260430_150405/groups/prod_abc")
    assert group.status_code == 200
    assert len(group.json()["offers"]) == 2

    explanation = client.get(
        "/runs/run_20260430_150405/explain",
        params={"source_id_a": "1", "source_id_b": "2"},
    )
    assert explanation.status_code == 200
    assert explanation.json()["found"] is True

