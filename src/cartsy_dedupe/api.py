from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from .artifact_index import search_artifacts
from .query import explain_pair, get_group, read_assignments, search_products
from .text import normalize_text


def create_app(*, runs_root: str | Path = "outputs") -> FastAPI:
    root = Path(os.getenv("CARTSY_RUNS_ROOT", str(runs_root)))
    app = FastAPI(
        title="Cartsy Dedupe API",
        version="0.1.0",
        description="REST interface for searching and inspecting Cartsy dedupe runs.",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "runs_root": str(root)}

    @app.get("/runs")
    def list_runs() -> dict[str, object]:
        runs = []
        if root.exists():
            for run_dir in sorted(root.iterdir(), reverse=True):
                if not run_dir.is_dir() or not (run_dir / "summary_report.json").exists():
                    continue
                summary = read_summary(run_dir)
                runs.append(
                    {
                        "run_id": summary.get("run_id", run_dir.name),
                        "path": str(run_dir),
                        "input_records": summary.get("input_records"),
                        "final_unique_products": summary.get("final_unique_products"),
                        "duplicate_records_grouped": summary.get("duplicate_records_grouped"),
                        "elapsed_seconds": summary.get("elapsed_seconds"),
                        "openai_cost_usd": summary.get("metrics", {}).get("openai", {}).get("total_estimated_cost_usd"),
                    }
                )
        return {"runs": runs}

    @app.get("/runs/{run_id}/summary")
    def get_summary(run_id: str) -> dict[str, object]:
        return read_summary(resolve_run(root, run_id))

    @app.get("/runs/{run_id}/products")
    def list_products(
        run_id: str,
        q: str | None = None,
        retailer: str | None = None,
        brand: str | None = None,
        dedupe_id: str | None = None,
        decision: str | None = None,
        min_confidence: float | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        run_dir = resolve_run(root, run_id)
        rows = filter_assignments(
            read_assignments(run_dir),
            q=q,
            retailer=retailer,
            brand=brand,
            dedupe_id=dedupe_id,
            decision=decision,
            min_confidence=min_confidence,
        )
        return {"total": len(rows), "limit": limit, "offset": offset, "products": rows[offset : offset + limit]}

    @app.get("/runs/{run_id}/search")
    def product_search(
        run_id: str,
        q: str,
        limit: int = Query(10, ge=1, le=100),
        backend: Literal["auto", "postgres", "artifacts"] = "auto",
    ) -> dict[str, object]:
        run_dir = resolve_run(root, run_id)
        try:
            results = search_products(run_dir, q, limit=limit, backend=backend)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"run_id": run_id, "query": q, "backend": backend, "results": results}

    @app.get("/runs/{run_id}/artifact-search")
    def artifact_search(
        run_id: str,
        q: str,
        type: Literal["group", "offer", "pair", "near_miss", "summary"] | None = None,
        limit: int = Query(10, ge=1, le=100),
    ) -> dict[str, object]:
        try:
            results = search_artifacts(q, run_id=run_id, artifact_type=type, limit=limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"run_id": run_id, "query": q, "type": type, "results": results}

    @app.get("/runs/{run_id}/groups/{dedupe_id}")
    def group_detail(run_id: str, dedupe_id: str) -> dict[str, object]:
        try:
            return get_group(resolve_run(root, run_id), dedupe_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/explain")
    def pair_explanation(run_id: str, source_id_a: str, source_id_b: str) -> dict[str, object]:
        return explain_pair(resolve_run(root, run_id), source_id_a, source_id_b)

    return app


def resolve_run(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run_dir


def read_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "summary_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"summary_report.json not found for {run_dir.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def filter_assignments(
    rows: list[dict[str, str]],
    *,
    q: str | None,
    retailer: str | None,
    brand: str | None,
    dedupe_id: str | None,
    decision: str | None,
    min_confidence: float | None,
) -> list[dict[str, str]]:
    query_norm = normalize_text(q)
    brand_norm = normalize_text(brand)
    filtered = []
    for row in rows:
        if retailer and row.get("retailer") != retailer:
            continue
        if dedupe_id and row.get("dedupe_id") != dedupe_id:
            continue
        if decision and row.get("decision") != decision:
            continue
        if brand_norm and brand_norm not in normalize_text(row.get("brand_raw", "") or row.get("canonical_brand", "")):
            continue
        if min_confidence is not None and safe_float(row.get("cluster_confidence")) < min_confidence:
            continue
        if query_norm:
            haystack = normalize_text(
                " ".join(
                    [
                        row.get("source_id", ""),
                        row.get("dedupe_id", ""),
                        row.get("retailer", ""),
                        row.get("name_raw", ""),
                        row.get("brand_raw", ""),
                        row.get("canonical_name", ""),
                        row.get("canonical_brand", ""),
                        row.get("sku", ""),
                        row.get("dimension", ""),
                    ]
                )
            )
            if query_norm not in haystack:
                continue
        filtered.append(row)
    return filtered


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
