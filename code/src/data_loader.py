"""
Deterministic lookups over the two small reference CSVs.

Why dict/DataFrame lookups and not a vector DB: evidence_requirements.csv
has 11 rows; user_history.csv has 47. Both are keyed by an exact field
(claim_object, user_id). A vector DB would replace an O(1) exact match with
approximate nearest-neighbor search over a problem that has exactly one
correct answer per key -- strictly worse on correctness, latency, and cost,
and it's an extra dependency to explain for no benefit. See README for the
interview-ready version of this argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


@dataclass(frozen=True)
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

    @property
    def history_flag_set(self) -> set[str]:
        if self.history_flags == "none":
            return set()
        return {f.strip() for f in self.history_flags.split(";") if f.strip()}


class ReferenceData:
    """Loads both reference CSVs once and exposes O(1) lookups."""

    def __init__(self, evidence_requirements_csv: Path, user_history_csv: Path):
        ev_df = pd.read_csv(evidence_requirements_csv)
        self._evidence_by_object: dict[str, list[EvidenceRequirement]] = {}
        for _, row in ev_df.iterrows():
            req = EvidenceRequirement(
                requirement_id=row["requirement_id"],
                claim_object=row["claim_object"],
                applies_to=row["applies_to"],
                minimum_image_evidence=row["minimum_image_evidence"],
            )
            self._evidence_by_object.setdefault(req.claim_object, []).append(req)

        hist_df = pd.read_csv(user_history_csv)
        self._history_by_user: dict[str, UserHistory] = {}
        for _, row in hist_df.iterrows():
            uh = UserHistory(
                user_id=row["user_id"],
                past_claim_count=int(row["past_claim_count"]),
                accept_claim=int(row["accept_claim"]),
                manual_review_claim=int(row["manual_review_claim"]),
                rejected_claim=int(row["rejected_claim"]),
                last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                history_flags=str(row["history_flags"]),
                history_summary=str(row["history_summary"]),
            )
            self._history_by_user[uh.user_id] = uh

    def evidence_for(self, claim_object: str) -> list[EvidenceRequirement]:
        """Rows that apply to this object: object-specific rows + 'all' rows."""
        specific = self._evidence_by_object.get(claim_object, [])
        general = self._evidence_by_object.get("all", [])
        return specific + general

    def history_for(self, user_id: str) -> UserHistory | None:
        return self._history_by_user.get(user_id)
