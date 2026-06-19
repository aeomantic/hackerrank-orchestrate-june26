"""
Deterministic merge of (a) the LLM's visual risk flags with (b) user
history, into the final risk_flags column.

This whole module exists because of one finding: joining
user_history.csv.history_flags against the labeled risk_flags column in
sample_claims.csv showed a clean, exceptionless rule across all 20 rows --

    "user_history_risk" in history_flags  =>  "user_history_risk" in output AND
                                               "manual_review_required" in output
    "manual_review_required" in history_flags (without user_history_risk)
                                          =>  "manual_review_required" in output

and separately, manual_review_required also appears whenever the image
itself is untrustworthy (valid_image=False) or the model raised a
trust-relevant visual flag (claim_mismatch, wrong_object,
possible_manipulation, non_original_image) -- even when history is clean.
Plain image-quality flags alone (blurry_image, wrong_angle,
low_light_or_glare) never triggered manual_review_required by themselves
in the sample set (case_003, case_006): those just mean "ask for a better
photo", not "this needs a human fraud/quality reviewer."

This is therefore implemented as plain Python, not as an LLM judgment call.
That is the direct, concrete answer to "how do you guarantee user history
never overrides clear visual evidence": history can only ever ADD entries
to this list. It has no code path that can change claim_status,
issue_type, object_part, or severity -- those are set once, upstream, by
the vision assessment and never touched again.
"""

from __future__ import annotations

from src.data_loader import UserHistory
from src.schemas import PrimaryAssessment

TRUST_RELEVANT_VISUAL_FLAGS = {
    "claim_mismatch",
    "wrong_object",
    "possible_manipulation",
    "non_original_image",
}


def merge_risk_flags(
    assessment: PrimaryAssessment,
    history: UserHistory | None,
    escalated_to_manual_review: bool = False,
) -> list[str]:
    """Pure function: visual flags + history -> final risk_flags list.

    `escalated_to_manual_review` is set by the pipeline's own circuit
    breaker (validation failures, unresolved self-consistency disagreement)
    -- a third, independent source of manual_review_required alongside
    history and visual trust flags.
    """
    flags: list[str] = list(assessment.visual_risk_flags)

    history_flag_set = history.history_flag_set if history else set()

    if "user_history_risk" in history_flag_set:
        flags.append("user_history_risk")

    needs_manual_review = (
        "manual_review_required" in history_flag_set
        or "user_history_risk" in history_flag_set
        or not assessment.valid_image
        or bool(TRUST_RELEVANT_VISUAL_FLAGS & set(assessment.visual_risk_flags))
        or escalated_to_manual_review
    )
    if needs_manual_review:
        flags.append("manual_review_required")

    # De-dupe while preserving first-seen order (model's own flags first,
    # then history-derived ones) -- purely cosmetic, doesn't affect logic.
    seen = set()
    deduped = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    return deduped if deduped else []
