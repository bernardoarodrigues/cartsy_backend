from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cartsy_dedupe.query import read_candidate_pairs


def evaluate_run_against_truth(
    *,
    run_dir: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path | None = None,
    include_blank_labels: bool = False,
    min_precision: float | None = None,
    min_recall: float | None = None,
    min_vector_only_precision: float | None = None,
) -> dict[str, object]:
    """Evaluate completed run candidate decisions against labeled source IDs.

    The evaluator intentionally scores the saved production candidate pairs, not
    random product-pair samples. This catches calibration failures that only
    appear after retrieval, such as overconfident vector-only candidates.
    """
    truth = read_ground_truth_labels(ground_truth_path, include_blank_labels=include_blank_labels)
    pairs = read_candidate_pairs(run_dir)
    overall = Confusion()
    slices: dict[str, Confusion] = {}
    false_positive_reasons: Counter[str] = Counter()
    false_negative_reasons: Counter[str] = Counter()
    unlabeled_pairs = 0

    def slice_confusion(name: str) -> Confusion:
        """Build slice confusion."""
        return slices.setdefault(name, Confusion())

    for pair in pairs:
        left_id = str(pair.get("product_a_id", ""))
        right_id = str(pair.get("product_b_id", ""))
        if left_id not in truth or right_id not in truth:
            unlabeled_pairs += 1
            continue

        expected_merge = truth[left_id] == truth[right_id]
        predicted_merge = str(pair.get("decision", "")) == "merge"
        overall.add(expected_merge=expected_merge, predicted_merge=predicted_merge)

        for name in pair_slice_names(pair):
            slice_confusion(name).add(expected_merge=expected_merge, predicted_merge=predicted_merge)

        if predicted_merge and not expected_merge:
            false_positive_reasons.update(reason_labels(pair))
        elif expected_merge and not predicted_merge:
            false_negative_reasons.update(reason_labels(pair))

    slice_metrics = {
        name: confusion.to_metrics()
        for name, confusion in sorted(
            slices.items(),
            key=lambda item: (-item[1].total, item[0]),
        )
    }
    report = {
        "run_dir": str(run_dir),
        "ground_truth_path": str(ground_truth_path),
        "include_blank_labels": include_blank_labels,
        "truth_labels": len(truth),
        "candidate_pairs_read": len(pairs),
        "labeled_candidate_pairs": overall.total,
        "unlabeled_candidate_pairs": unlabeled_pairs,
        "overall": overall.to_metrics(),
        "slices": slice_metrics,
        "false_positive_reasons": dict(false_positive_reasons.most_common(30)),
        "false_negative_reasons": dict(false_negative_reasons.most_common(30)),
    }
    report["acceptance"] = acceptance_report(
        overall=report["overall"],
        slices=slice_metrics,
        min_precision=min_precision,
        min_recall=min_recall,
        min_vector_only_precision=min_vector_only_precision,
    )
    if output_path is not None:
        resolved_output = Path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def acceptance_report(
    *,
    overall: dict[str, object],
    slices: dict[str, dict[str, object]],
    min_precision: float | None,
    min_recall: float | None,
    min_vector_only_precision: float | None,
) -> dict[str, object]:
    """Build pass/fail acceptance checks for labeled evaluation metrics."""
    checks: list[dict[str, object]] = []
    if min_precision is not None:
        checks.append(metric_check("overall.precision", overall.get("precision"), min_precision))
    if min_recall is not None:
        checks.append(metric_check("overall.recall", overall.get("recall"), min_recall))
    if min_vector_only_precision is not None:
        vector_metrics = slices.get("risk:vector_only")
        no_vector_merges = bool(vector_metrics is not None and int(vector_metrics.get("predicted_merge_pairs") or 0) == 0)
        checks.append(
            metric_check(
                "risk:vector_only.precision",
                None if vector_metrics is None else vector_metrics.get("precision"),
                min_vector_only_precision,
                missing_passes=vector_metrics is None or no_vector_merges,
            )
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def metric_check(
    name: str,
    value: object,
    threshold: float,
    *,
    missing_passes: bool = False,
) -> dict[str, object]:
    """Build metric check."""
    numeric_value = coerce_optional_float(value)
    passed = missing_passes if numeric_value is None else numeric_value >= threshold
    return {
        "name": name,
        "value": numeric_value,
        "threshold": threshold,
        "passed": passed,
    }


class Confusion:
    """Mutable confusion-matrix counts for one evaluation slice."""
    def __init__(self) -> None:
        """Initialize the object state used by this component."""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @property
    def total(self) -> int:
        """Return the total number of evaluated pairs."""
        return self.tp + self.fp + self.fn + self.tn

    def add(self, *, expected_merge: bool, predicted_merge: bool) -> None:
        """Add counts or values into the accumulator."""
        if predicted_merge and expected_merge:
            self.tp += 1
        elif predicted_merge and not expected_merge:
            self.fp += 1
        elif not predicted_merge and expected_merge:
            self.fn += 1
        else:
            self.tn += 1

    def to_metrics(self) -> dict[str, object]:
        """Convert confusion counts into precision, recall, and F1 metrics."""
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        return {
            "pairs": self.total,
            "truth_positive_pairs": self.tp + self.fn,
            "truth_negative_pairs": self.fp + self.tn,
            "predicted_merge_pairs": self.tp + self.fp,
            "predicted_no_merge_pairs": self.fn + self.tn,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
        }


def read_ground_truth_labels(path: str | Path, *, include_blank_labels: bool = False) -> dict[str, str]:
    """Read labeled source-to-deduped-id rows from ground truth CSV."""
    labels: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_id = str(row.get("source_id", "")).strip()
            deduped_id = str(row.get("deduped_id", "")).strip()
            if not source_id:
                continue
            if not deduped_id and not include_blank_labels:
                continue
            labels[source_id] = deduped_id
    return labels


def pair_slice_names(pair: dict[str, Any]) -> list[str]:
    """Build pair slice names."""
    layer_names = retrieval_layers(pair)
    names = ["all_labeled"]
    if layer_names:
        names.append("layers:" + "+".join(layer_names))
    else:
        names.append("layers:none")
    if layer_names == ["vector"]:
        names.append("risk:vector_only")
    if "exact" in layer_names:
        names.append("evidence:exact")
    if "lexical" in layer_names:
        names.append("evidence:lexical")
    if "trigram" in layer_names:
        names.append("evidence:trigram")
    if "vector" in layer_names:
        names.append("evidence:vector")
    explanation = str(pair.get("explanation", ""))
    if "generic_brand" in explanation:
        names.append("risk:generic_brand")
    if "below_evidence_threshold" in explanation:
        names.append("decision:below_evidence_threshold")
    return names


def retrieval_layers(pair: dict[str, Any]) -> list[str]:
    """Extract retrieval layers."""
    blocking_keys = str(pair.get("blocking_keys", ""))
    layers: list[str] = []
    for prefix, name in (
        ("exact:", "exact"),
        ("lexical:", "lexical"),
        ("trigram:", "trigram"),
        ("vector:", "vector"),
    ):
        if prefix in blocking_keys:
            layers.append(name)
    return layers


def reason_labels(pair: dict[str, Any]) -> list[str]:
    """Extract reason labels."""
    labels: list[str] = []
    decision_reason = str(pair.get("decision_reason", "")).strip()
    if decision_reason:
        labels.append(f"decision_reason:{decision_reason}")
    explanation = str(pair.get("explanation", ""))
    for part in explanation.split("; "):
        if not part:
            continue
        key = part.split(":", 1)[0]
        if key and key not in labels:
            labels.append(key)
    return labels


def safe_div(numerator: float, denominator: float) -> float | None:
    """Safely compute safe div."""
    if denominator == 0:
        return None
    return numerator / denominator


def coerce_optional_float(value: object) -> float | None:
    """Coerce optional numeric artifact values to floats."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
