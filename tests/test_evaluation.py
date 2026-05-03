from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl

from cartsy_dedupe.cli import main
from cartsy_dedupe.evaluation import evaluate_run_against_truth


def write_truth(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "deduped_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"source_id": "1", "deduped_id": "a"},
                {"source_id": "2", "deduped_id": "a"},
                {"source_id": "3", "deduped_id": "b"},
                {"source_id": "4", "deduped_id": "c"},
                {"source_id": "5", "deduped_id": ""},
            ]
        )


def write_run_pairs(run_dir: Path) -> None:
    run_dir.mkdir()
    pl.DataFrame(
        [
            {
                "product_a_id": "1",
                "product_b_id": "2",
                "score": 0.92,
                "ml_score": 0.96,
                "evidence_score": 0.92,
                "decision_threshold": 0.84,
                "decision_reason": "ml_score_above_threshold",
                "decision": "merge",
                "explanation": "relation:candidate_match; decision_reason:ml_score_above_threshold; title_high",
                "blocking_keys": "lexical:fts:0.7143|trigram:title:1.0000",
                "feature_scores": "",
            },
            {
                "product_a_id": "3",
                "product_b_id": "4",
                "score": 0.43,
                "ml_score": 0.96,
                "evidence_score": 0.43,
                "decision_threshold": 0.84,
                "decision_reason": "ml_score_above_threshold",
                "decision": "merge",
                "explanation": "relation:candidate_match; decision_reason:ml_score_above_threshold; generic_brand",
                "blocking_keys": "vector:cosine:0.7806",
                "feature_scores": "",
            },
            {
                "product_a_id": "1",
                "product_b_id": "3",
                "score": 0.61,
                "ml_score": 0.91,
                "evidence_score": 0.61,
                "decision_threshold": 0.84,
                "decision_reason": "below_evidence_threshold",
                "decision": "no_merge",
                "explanation": "relation:similar_related_product; decision_reason:below_evidence_threshold",
                "blocking_keys": "vector:cosine:0.7900",
                "feature_scores": "",
            },
            {
                "product_a_id": "4",
                "product_b_id": "5",
                "score": 0.99,
                "ml_score": 1.0,
                "evidence_score": 0.99,
                "decision_threshold": 0.84,
                "decision_reason": "rule_certain_match",
                "decision": "merge",
                "explanation": "relation:certain_match",
                "blocking_keys": "exact:ean:123",
                "feature_scores": "",
            },
        ]
    ).write_parquet(run_dir / "candidate_pairs.parquet")


def test_evaluate_run_against_truth_reports_overall_and_risky_slices(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    truth_path = tmp_path / "truth.csv"
    output_path = tmp_path / "eval.json"
    write_run_pairs(run_dir)
    write_truth(truth_path)

    report = evaluate_run_against_truth(
        run_dir=run_dir,
        ground_truth_path=truth_path,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert report["candidate_pairs_read"] == 4
    assert report["labeled_candidate_pairs"] == 3
    assert report["unlabeled_candidate_pairs"] == 1
    assert report["overall"]["precision"] == 0.5
    assert report["overall"]["recall"] == 1.0
    assert report["slices"]["risk:vector_only"]["fp"] == 1
    assert report["slices"]["risk:generic_brand"]["fp"] == 1
    assert report["false_positive_reasons"]["decision_reason:ml_score_above_threshold"] == 1
    assert report["acceptance"]["passed"] is True


def test_evaluate_run_acceptance_checks_fail_when_precision_is_low(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    truth_path = tmp_path / "truth.csv"
    write_run_pairs(run_dir)
    write_truth(truth_path)

    report = evaluate_run_against_truth(
        run_dir=run_dir,
        ground_truth_path=truth_path,
        min_precision=0.90,
        min_recall=0.90,
        min_vector_only_precision=0.50,
    )

    checks = {check["name"]: check for check in report["acceptance"]["checks"]}
    assert report["acceptance"]["passed"] is False
    assert checks["overall.precision"]["passed"] is False
    assert checks["overall.recall"]["passed"] is True
    assert checks["risk:vector_only.precision"]["passed"] is False


def test_evaluate_run_cli_writes_default_report(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    truth_path = tmp_path / "truth.csv"
    write_run_pairs(run_dir)
    write_truth(truth_path)

    result = main(["evaluate-run", "--run", str(run_dir), "--ground-truth", str(truth_path)])

    assert result == 0
    report_path = run_dir / "labeled_evaluation.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["overall"]["fp"] == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["overall"]["tp"] == 1


def test_evaluate_run_cli_returns_nonzero_when_acceptance_fails(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    truth_path = tmp_path / "truth.csv"
    write_run_pairs(run_dir)
    write_truth(truth_path)

    result = main(
        [
            "evaluate-run",
            "--run",
            str(run_dir),
            "--ground-truth",
            str(truth_path),
            "--min-precision",
            "0.90",
        ]
    )

    assert result == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["acceptance"]["passed"] is False
