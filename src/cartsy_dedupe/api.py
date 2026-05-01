from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .artifact_index import search_artifacts
from .query import (
    build_group_graph,
    explain_pair,
    get_group,
    read_assignments,
    read_candidate_pairs,
    read_groups,
    read_near_misses,
    search_products,
)
from .text import normalize_text


def create_app(*, runs_root: str | Path = "outputs") -> FastAPI:
    root = Path(os.getenv("CARTSY_RUNS_ROOT", str(runs_root)))
    app = FastAPI(
        title="Cartsy Dedupe API",
        version="0.1.0",
        description="REST interface for searching and inspecting Cartsy dedupe runs.",
    )
    cors_origins = os.getenv("CARTSY_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins if origin.strip()] or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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

    @app.get("/runs/{run_id}/groups")
    def list_groups(
        run_id: str,
        q: str | None = None,
        retailer: str | None = None,
        min_offers: int | None = None,
        min_confidence: float | None = None,
        sort: Literal["num_offers", "cluster_confidence", "dedupe_id"] = "num_offers",
        order: Literal["asc", "desc"] = "desc",
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        run_dir = resolve_run(root, run_id)
        groups = read_groups(run_dir)
        query_norm = normalize_text(q)
        filtered: list[dict[str, object]] = []
        for group in groups:
            if min_offers is not None and int(group.get("num_offers") or 0) < min_offers:
                continue
            if min_confidence is not None and safe_float(group.get("cluster_confidence")) < min_confidence:
                continue
            if retailer and retailer not in (group.get("retailers") or []):
                continue
            if query_norm:
                hay = normalize_text(
                    " ".join(
                        [
                            str(group.get("dedupe_id", "")),
                            str(group.get("canonical_name", "")),
                            str(group.get("canonical_brand", "")),
                            str(group.get("canonical_category", "")),
                        ]
                    )
                )
                if query_norm not in hay:
                    continue
            filtered.append(group)

        reverse = order == "desc"
        if sort == "num_offers":
            filtered.sort(key=lambda g: int(g.get("num_offers") or 0), reverse=reverse)
        elif sort == "cluster_confidence":
            filtered.sort(key=lambda g: safe_float(g.get("cluster_confidence")), reverse=reverse)
        else:
            filtered.sort(key=lambda g: str(g.get("dedupe_id") or ""), reverse=reverse)

        return {
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "groups": filtered[offset : offset + limit],
        }

    @app.get("/runs/{run_id}/groups/{dedupe_id}")
    def group_detail(run_id: str, dedupe_id: str) -> dict[str, object]:
        try:
            return get_group(resolve_run(root, run_id), dedupe_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/groups/{dedupe_id}/graph")
    def group_graph(run_id: str, dedupe_id: str) -> dict[str, object]:
        try:
            return build_group_graph(resolve_run(root, run_id), dedupe_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/near-misses")
    def list_near_misses(
        run_id: str,
        q: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        run_dir = resolve_run(root, run_id)
        rows = read_near_misses(run_dir)
        query_norm = normalize_text(q)
        out: list[dict[str, str]] = []
        for row in rows:
            score = safe_float(row.get("score"))
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
            if query_norm:
                hay = normalize_text(
                    " ".join(
                        [
                            row.get("name_a", ""),
                            row.get("name_b", ""),
                            row.get("brand_a", ""),
                            row.get("brand_b", ""),
                            row.get("retailer_a", ""),
                            row.get("retailer_b", ""),
                            row.get("explanation", ""),
                        ]
                    )
                )
                if query_norm not in hay:
                    continue
            out.append(row)
        return {"total": len(out), "limit": limit, "offset": offset, "pairs": out[offset : offset + limit]}

    @app.get("/runs/{run_id}/pairs")
    def list_pairs(
        run_id: str,
        decision: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        source_id: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, object]:
        run_dir = resolve_run(root, run_id)
        rows = read_candidate_pairs(run_dir)
        out: list[dict[str, object]] = []
        for row in rows:
            score = safe_float(row.get("score"))
            if decision and row.get("decision") != decision:
                continue
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
            if source_id and source_id not in {str(row.get("product_a_id", "")), str(row.get("product_b_id", ""))}:
                continue
            out.append(row)
        out.sort(key=lambda r: safe_float(r.get("score")), reverse=True)
        return {"total": len(out), "limit": limit, "offset": offset, "pairs": out[offset : offset + limit]}

    @app.get("/runs/{run_id}/dataset-graph")
    def dataset_graph(
        run_id: str,
        min_group_size: int = Query(2, ge=1, le=100),
        include_singletons: bool = False,
        max_groups: int = Query(300, ge=1, le=5000),
        max_singletons: int = Query(0, ge=0, le=5000),
    ) -> dict[str, object]:
        run_dir = resolve_run(root, run_id)
        assignments = read_assignments(run_dir)

        # Bucket by dedupe_id
        members_by_group: dict[str, list[dict[str, str]]] = {}
        for row in assignments:
            members_by_group.setdefault(row["dedupe_id"], []).append(row)

        # Sort multi-product groups by size (descending)
        multi_groups = [
            (gid, members)
            for gid, members in members_by_group.items()
            if len(members) >= min_group_size and len(members) > 1
        ]
        multi_groups.sort(key=lambda item: len(item[1]), reverse=True)
        truncated_groups = len(multi_groups) > max_groups
        multi_groups = multi_groups[:max_groups]

        kept_member_ids: set[str] = set()
        nodes: list[dict[str, object]] = []
        for gid, members in multi_groups:
            for m in members:
                kept_member_ids.add(m["source_id"])
                nodes.append(
                    {
                        "id": m["source_id"],
                        "source_id": m["source_id"],
                        "dedupe_id": gid,
                        "group_size": len(members),
                        "name": m.get("name_raw", ""),
                        "brand": m.get("brand_raw", ""),
                        "retailer": m.get("retailer", ""),
                        "price_cents": m.get("price_cents", ""),
                        "decision": m.get("decision", ""),
                        "is_singleton": False,
                    }
                )

        singleton_total = 0
        if include_singletons:
            singletons = [m[0] for gid, m in members_by_group.items() if len(m) == 1]
            singleton_total = len(singletons)
            for m in singletons[:max_singletons]:
                nodes.append(
                    {
                        "id": m["source_id"],
                        "source_id": m["source_id"],
                        "dedupe_id": m["dedupe_id"],
                        "group_size": 1,
                        "name": m.get("name_raw", ""),
                        "brand": m.get("brand_raw", ""),
                        "retailer": m.get("retailer", ""),
                        "price_cents": m.get("price_cents", ""),
                        "decision": m.get("decision", ""),
                        "is_singleton": True,
                    }
                )

        # Edges: only merge decisions where both endpoints are in our node set
        edges: list[dict[str, object]] = []
        for pair in read_candidate_pairs(run_dir):
            if pair.get("decision") != "merge":
                continue
            a = str(pair.get("product_a_id", ""))
            b = str(pair.get("product_b_id", ""))
            if a in kept_member_ids and b in kept_member_ids:
                edges.append(
                    {
                        "source": a,
                        "target": b,
                        "score": float(pair.get("score") or 0.0),
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "groups_returned": len(multi_groups),
                "groups_truncated": truncated_groups,
                "singletons_total": singleton_total,
                "singletons_returned": min(singleton_total, max_singletons) if include_singletons else 0,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

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
