"""
Scoring functions for comparing predictions against the labeled
sample_claims.csv. Imported by evaluation/main.py; not a standalone CLI.

Per the problem statement's own guidance ("exact_match plus precision/recall
is sufficient and shows you understand the metrics"), this deliberately
does NOT build an LLM-as-judge or a RAGAS-style pipeline -- both would be
disproportionate machinery for a 20-row labeled set, and would themselves
need defending in the interview as "why did you build an eval system to
evaluate your eval system."
"""

from __future__ import annotations

import csv
from pathlib import Path

CATEGORICAL_FIELDS = ["claim_status", "issue_type", "object_part", "severity"]
BOOLEAN_FIELDS = ["evidence_standard_met", "valid_image"]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


def load_rows(path: Path) -> dict[str, dict]:
    """Keyed by (user_id, image_paths) since user_id alone isn't unique
    across rows in claims.csv (a few users file more than once)."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["user_id"], r["image_paths"]): r for r in rows}


def set_from_field(value: str) -> set[str]:
    if value.strip().lower() == "none":
        return set()
    return {v.strip() for v in value.split(";") if v.strip()}


def evaluate_rows(preds: dict[tuple, dict], truth: dict[tuple, dict]) -> dict:
    """Same scoring as evaluate(), but takes already-loaded row dicts keyed
    by (user_id, image_paths) -- lets evaluation/main.py score in-memory
    pipeline output without a round-trip through disk."""
    keys = [k for k in truth if k in preds]
    missing = [k for k in truth if k not in preds]

    results = {"n_ground_truth": len(truth), "n_matched": len(keys), "n_missing_predictions": len(missing)}

    for field in CATEGORICAL_FIELDS + BOOLEAN_FIELDS:
        correct = sum(1 for k in keys if str(preds[k][field]).strip().lower() == str(truth[k][field]).strip().lower())
        results[f"{field}_accuracy"] = round(correct / len(keys), 3) if keys else None

    # claim_status precision/recall/F1 per class (the field the problem
    # statement cares about most -- it's the actual decision).
    classes = ["supported", "contradicted", "not_enough_information"]
    per_class = {}
    for cls in classes:
        tp = sum(1 for k in keys if preds[k]["claim_status"] == cls and truth[k]["claim_status"] == cls)
        fp = sum(1 for k in keys if preds[k]["claim_status"] == cls and truth[k]["claim_status"] != cls)
        fn = sum(1 for k in keys if preds[k]["claim_status"] != cls and truth[k]["claim_status"] == cls)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None
        per_class[cls] = {
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "support": sum(1 for k in keys if truth[k]["claim_status"] == cls),
        }
    results["claim_status_per_class"] = per_class

    # Set-overlap (Jaccard) for risk_flags and supporting_image_ids -- exact
    # string match is too strict for multi-valued semicolon fields.
    for field in SET_FIELDS:
        jaccards = []
        for k in keys:
            a, b = set_from_field(str(preds[k][field])), set_from_field(str(truth[k][field]))
            if not a and not b:
                jaccards.append(1.0)
            else:
                jaccards.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        results[f"{field}_mean_jaccard"] = round(sum(jaccards) / len(jaccards), 3) if jaccards else None

    return results


def evaluate(predictions_path: Path, ground_truth_path: Path) -> dict:
    """File-based convenience wrapper around evaluate_rows()."""
    return evaluate_rows(load_rows(predictions_path), load_rows(ground_truth_path))
