"""
Pydantic schemas for the claim review pipeline.

These schemas are the single source of truth for what a valid model
response looks like. Every constraint here is either:
  (a) given directly by the problem statement's allowed-value lists, or
  (b) reverse-engineered from patterns observed in dataset/sample_claims.csv
      (see comments next to each validator for the specific evidence).

Design note: we deliberately do NOT ask the LLM to output
`user_history_risk` or `manual_review_required` directly. Joining
user_history.csv.history_flags against sample_claims.csv.risk_flags showed
those two flags follow a clean, deterministic rule (see src/risk_merge.py).
Computing them in plain Python instead of asking the model to "remember"
to add them removes an entire class of omission errors and is far easier
to defend than "the model decided to add it."
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums (Literal types). Kept as flat Literals rather than Python Enums --
# Pydantic validates these identically, and Literal reads directly off the
# problem statement's "Allowed values" section with zero translation layer.
# ---------------------------------------------------------------------------

ClaimObject = Literal["car", "laptop", "package"]

ClaimStatus = Literal["supported", "contradicted", "not_enough_information"]

IssueType = Literal[
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain",
    "none", "unknown",
]

# Flat union of every object_part value across all three object types, plus
# "unknown". We validate "is this part legal for this claim_object" with a
# cross-field validator against OBJECT_PART_ALLOWLIST below, rather than
# three separate enums -- one Literal + one dict is simpler to explain than
# a discriminated union, and the validator is the same handful of lines
# regardless of object type.
ObjectPart = Literal[
    # car
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body",
    # laptop
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
    # package
    "box", "package_corner", "package_side", "seal", "label", "contents", "item",
    # shared
    "unknown",
]

OBJECT_PART_ALLOWLIST: dict[ClaimObject, set[str]] = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

Severity = Literal["none", "low", "medium", "high", "unknown"]

# Risk flags the MODEL is allowed to emit directly -- all visually/textually
# grounded. user_history_risk and manual_review_required are intentionally
# excluded: they are computed deterministically in risk_merge.py.
VisualRiskFlag = Literal[
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present",
]

# Full output vocabulary (used when validating the final merged row).
AllRiskFlag = Literal[
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]


# ---------------------------------------------------------------------------
# Per-image scratchpad observation.
#
# This is the "grounding" step: the model must describe what it sees BEFORE
# it is allowed to classify. This is the primary hallucination defense --
# see prompts.py for why this ordering matters.
# ---------------------------------------------------------------------------

class ImageObservation(BaseModel):
    image_id: str = Field(description="Filename without extension, e.g. 'img_1'.")
    object_type_matches_claim: bool = Field(
        description="True if this image plausibly shows the claimed object "
        "type (car/laptop/package) at all -- independent of damage."
    )
    relevant_part_visible: bool = Field(
        description="True if the part relevant to the claim is visible "
        "clearly enough to assess its condition."
    )
    observed_condition: str = Field(
        description="Brief, concrete description of what is actually visible "
        "(e.g. 'small scratch on rear bumper corner', 'mirror and road, no "
        "headlight in frame'). Must describe pixels, not assumptions."
    )
    quality_or_trust_notes: list[str] = Field(
        default_factory=list,
        description="Free-text notes on anything affecting usability or "
        "trust: blur, glare, cropping, visible watermark/stock-photo marks, "
        "staged/unrealistic scene, sticky notes or handwriting embedded in "
        "the photo, signs of editing, etc. Empty list if none.",
    )


# ---------------------------------------------------------------------------
# Primary model output (one LLM call produces this).
# ---------------------------------------------------------------------------

class PrimaryAssessment(BaseModel):
    image_observations: list[ImageObservation]

    evidence_standard_met: bool
    evidence_standard_met_reason: str = Field(max_length=400)

    issue_type: IssueType
    object_part: ObjectPart

    claim_status: ClaimStatus
    claim_status_justification: str = Field(max_length=500)

    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="Image IDs grounding the decision (whichever decision "
        "that is -- supporting a 'contradicted' verdict counts). Empty list "
        "if no image is sufficient.",
    )

    visual_risk_flags: list[VisualRiskFlag] = Field(default_factory=list)

    valid_image: bool = Field(
        description="True if this image set is trustworthy/usable for "
        "automated review. A real photo that simply shows the wrong angle "
        "is still valid_image=True (see prompts.py for the worked "
        "examples this distinction is based on)."
    )

    severity: Severity

    confidence: float = Field(ge=0.0, le=1.0)

    # -- Cross-field validators -------------------------------------------
    # Each of these encodes a pattern that held across all 20 labeled rows
    # in sample_claims.csv. None of these are guesses.

    @model_validator(mode="after")
    def _supporting_ids_match_status(self) -> "PrimaryAssessment":
        # Verified across all 20 sample rows: supporting_image_ids is
        # non-empty iff claim_status is supported or contradicted, and
        # empty iff not_enough_information.
        has_support = len(self.supporting_image_ids) > 0
        needs_support = self.claim_status in ("supported", "contradicted")
        if has_support != needs_support:
            raise ValueError(
                f"claim_status={self.claim_status!r} requires "
                f"supporting_image_ids to be {'non-empty' if needs_support else 'empty'}, "
                f"got {self.supporting_image_ids!r}"
            )
        return self

    @model_validator(mode="after")
    def _none_issue_implies_none_severity(self) -> "PrimaryAssessment":
        # Verified: issue_type == "none" co-occurs with severity == "none"
        # in every sample row (case_014, case_020). The converse direction
        # (severity == "none" => issue_type == "none") also held, so this
        # is a true biconditional in the observed data.
        if (self.issue_type == "none") != (self.severity == "none"):
            raise ValueError(
                "issue_type='none' must pair with severity='none' and vice "
                f"versa; got issue_type={self.issue_type!r} severity={self.severity!r}"
            )
        return self

    @model_validator(mode="after")
    def _supported_has_real_issue(self) -> "PrimaryAssessment":
        # Verified: no "supported" row in the sample set has issue_type
        # "none". A supported claim always points at a real, visible issue.
        if self.claim_status == "supported" and self.issue_type == "none":
            raise ValueError("claim_status='supported' cannot pair with issue_type='none'")
        return self

    # Deliberately NOT enforced (see README "Validation rules we rejected"):
    # issue_type == "unknown" => severity == "unknown". case_033 in the
    # sample set is wrong_object with issue_type=unknown but severity=low --
    # there is visible damage on the (wrong) object, so severity is still
    # assessable even though the issue can't be named within the claimed
    # object's taxonomy. A hard rule here would reject a real labeled example.


class PrimaryAssessmentContext(BaseModel):
    """Validation-time context, not part of the LLM's output schema.

    Used by the pipeline to check supporting_image_ids and object_part
    against the actual submitted images / claim_object for this row.
    """
    submitted_image_ids: list[str]
    claim_object: ClaimObject


def validate_against_context(
    assessment: PrimaryAssessment, ctx: PrimaryAssessmentContext
) -> list[str]:
    """Returns a list of validation error strings (empty list = valid).

    Kept separate from the Pydantic model itself because these checks need
    per-row context (which images were actually submitted, which object was
    claimed) that isn't part of the LLM's output and shouldn't be re-stated
    by the model on every call.
    """
    errors: list[str] = []

    submitted = set(ctx.submitted_image_ids)
    hallucinated_ids = set(assessment.supporting_image_ids) - submitted
    if hallucinated_ids:
        errors.append(
            f"supporting_image_ids references IDs not in the submitted set: "
            f"{sorted(hallucinated_ids)} (submitted: {sorted(submitted)})"
        )

    observed_ids = {obs.image_id for obs in assessment.image_observations}
    hallucinated_obs = observed_ids - submitted
    if hallucinated_obs:
        errors.append(
            f"image_observations references IDs not in the submitted set: "
            f"{sorted(hallucinated_obs)}"
        )

    allowed_parts = OBJECT_PART_ALLOWLIST[ctx.claim_object]
    if assessment.object_part not in allowed_parts:
        errors.append(
            f"object_part={assessment.object_part!r} is not valid for "
            f"claim_object={ctx.claim_object!r} (allowed: {sorted(allowed_parts)})"
        )

    return errors


# ---------------------------------------------------------------------------
# Final output row -- exact column order required by the problem statement.
# ---------------------------------------------------------------------------

class OutputRow(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: ClaimObject
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str  # semicolon-joined, or "none"
    issue_type: IssueType
    object_part: ObjectPart
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: str  # semicolon-joined, or "none"
    valid_image: bool
    severity: Severity

    @field_validator("risk_flags", "supporting_image_ids", mode="before")
    @classmethod
    def _empty_list_to_none_string(cls, v):
        if isinstance(v, list):
            return ";".join(v) if v else "none"
        return v


# Module-level constant (NOT a model field -- kept outside the class body so
# Pydantic doesn't try to treat it as part of the schema).
OUTPUT_COLUMN_ORDER: list[str] = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]
