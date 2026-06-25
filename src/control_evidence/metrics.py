from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import AssessmentCase, AssessmentResult, Status


def classification_metrics(cases: list[AssessmentCase], results: list[AssessmentResult]) -> dict[str, Any]:
    gold = {case.case_id: case.gold_status for case in cases}
    pairs = [(gold[result.case_id], result.status) for result in results if gold[result.case_id] is not None]
    correct = sum(expected == predicted for expected, predicted in pairs)
    labels = list(Status)
    f1s: list[float] = []
    per_status: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(expected == label and predicted == label for expected, predicted in pairs)
        fp = sum(expected != label and predicted == label for expected, predicted in pairs)
        fn = sum(expected == label and predicted != label for expected, predicted in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if any(expected == label for expected, _ in pairs):
            f1s.append(f1)
        per_status[label.value] = {
            "support": sum(expected == label for expected, _ in pairs),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "n_cases": len(pairs),
        "accuracy": round(correct / len(pairs), 6) if pairs else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "per_status": per_status,
        "predicted_counts": dict(Counter(result.status.value for result in results)),
    }
