"""
Per-claim state machine.

    1. Build deterministic context (evidence checklist only -- NOT user
       history; see prompts.py for why).
    2. Primary vision pass.
    3. Validate. On failure: retry once with the exact error injected.
       Second failure: hard fallback (not_enough_information,
       manual_review_required), no further LLM calls for this row.
    4. Confidence triage:
         < CONFIDENCE_FLOOR        -> treat as already-failed, skip to
                                       fallback without spending a second
                                       call (the model is telling us it
                                       doesn't know; spending more tokens
                                       won't fix that).
         CONFIDENCE_FLOOR..CEILING,
         or a trust-relevant visual
         flag present                -> "borderline": run an independent
                                       second pass (step 5).
         >= CONFIDENCE_CEILING      -> accept the primary pass directly.
    5. Self-consistency pass (only if borderline): same model, a small
       nonzero temperature so it's a genuinely independent sample rather
       than a near-duplicate of a temperature=0 call. Compare claim_status
       and issue_type against pass 1.
         agree    -> accept pass 1, done.
         disagree -> step 6.
    6. Escalation tie-break (only on actual disagreement -- this is what
       keeps it rare and therefore cheap): one call to a stronger model,
       shown both prior verdicts, asked to adjudicate or declare the case
       irresolvable. If it still can't resolve confidently, hard fallback.
    7. Deterministic risk merge (src/risk_merge.py) -- the only place
       user_history.csv enters the computation, and only additively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from src.data_loader import ReferenceData
from src.image_utils import load_images_for_call, parse_image_paths, image_id_from_path
from src.llm_client import LLMClient, PRIMARY_MODEL, ESCALATION_MODEL
from src.prompts import SYSTEM_PROMPT, build_user_prompt, build_escalation_prompt
from src.risk_merge import merge_risk_flags
from src.schemas import (
    PrimaryAssessment,
    PrimaryAssessmentContext,
    OutputRow,
    validate_against_context,
)

CONFIDENCE_FLOOR = 0.4
CONFIDENCE_CEILING = 0.75
SELF_CONSISTENCY_TEMPERATURE = 0.4
PRIMARY_TEMPERATURE = 0.0
MAX_VALIDATION_RETRIES = 1

TRUST_RELEVANT_VISUAL_FLAGS = {
    "claim_mismatch", "wrong_object", "possible_manipulation", "non_original_image",
}


@dataclass
class ClaimInput:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str


@dataclass
class PipelineTrace:
    """Everything that happened while processing one row -- not written to
    output.csv, but logged/returned for the evaluation report and for
    debugging. This is the audit trail an interviewer would ask to see."""
    validation_retries: int = 0
    hard_fallback: bool = False
    hard_fallback_reason: str | None = None
    ran_self_consistency: bool = False
    self_consistency_agreed: bool | None = None
    ran_escalation: bool = False
    primary_confidence: float | None = None
    model_calls: int = 0


def _get_validated_assessment(
    client: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    images: list[dict],
    temperature: float,
    model: str,
    ctx: PrimaryAssessmentContext,
    trace: PipelineTrace,
) -> PrimaryAssessment | None:
    """Calls the LLM, validates against schema + context, retries once on
    failure with the error injected, returns None if still invalid."""
    retry_context = None
    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        trace.model_calls += 1
        raw = client.get_assessment(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            temperature=temperature,
            model=model,
            retry_context=retry_context,
        )
        errors: list[str] = []
        assessment: PrimaryAssessment | None = None
        try:
            assessment = PrimaryAssessment.model_validate(raw)
        except ValidationError as e:
            errors = [str(e)]
        if assessment is not None:
            errors = validate_against_context(assessment, ctx)
        if not errors:
            return assessment
        retry_context = "; ".join(errors)
        trace.validation_retries += 1
    return None


def _hard_fallback_row(claim: ClaimInput, reason: str) -> dict:
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": f"Automated review could not produce a valid assessment ({reason}).",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Automated review failed validation twice and was routed to manual review.",
        "supporting_image_ids": [],
        "visual_risk_flags": [],
        "valid_image": False,
        "severity": "unknown",
    }


def build_error_fallback_row(claim: ClaimInput, reference_data, error: str) -> OutputRow:
    """Used when process_claim itself raises an unhandled exception
    (unreadable image, unexpected error) rather than failing validation
    gracefully. Guarantees one bad row can never crash the whole batch."""
    fallback = _hard_fallback_row(claim, error)
    return _finalize(claim, fallback, reference_data, escalation_triggered=True)


def _summarize(assessment: PrimaryAssessment) -> str:
    return (
        f"claim_status={assessment.claim_status}, issue_type={assessment.issue_type}, "
        f"object_part={assessment.object_part}, confidence={assessment.confidence:.2f}. "
        f"Justification: {assessment.claim_status_justification}"
    )


def process_claim(
    claim: ClaimInput,
    *,
    dataset_root: Path,
    reference_data: ReferenceData,
    primary_client: LLMClient,
    escalation_client: LLMClient | None = None,
    enable_self_consistency: bool = True,
) -> tuple[OutputRow, PipelineTrace]:
    """
    `enable_self_consistency=False` collapses the pipeline to "Strategy A"
    in evaluation/main.py: a single primary pass, accepted as-is regardless
    of confidence (still validated/retried -- that circuit breaker is never
    optional). `enable_self_consistency=True` (default) is "Strategy B",
    the full pipeline described in the README. Both are real, runnable
    configurations of the same code, not two different implementations --
    this is what lets evaluation/main.py compare them head-to-head instead
    of just asserting one is better.
    """
    trace = PipelineTrace()
    escalation_client = escalation_client or primary_client

    submitted_paths = parse_image_paths(claim.image_paths)
    submitted_ids = [image_id_from_path(p) for p in submitted_paths]
    ctx = PrimaryAssessmentContext(submitted_image_ids=submitted_ids, claim_object=claim.claim_object)

    # Step 0: no images submitted at all -- trivial deterministic fallback,
    # no LLM call needed or warranted.
    if not submitted_ids:
        fallback = _hard_fallback_row(claim, "no images submitted")
        return _finalize(claim, fallback, reference_data, escalation_triggered=False), trace

    images = load_images_for_call(dataset_root, claim.image_paths)
    evidence_reqs = reference_data.evidence_for(claim.claim_object)
    user_prompt = build_user_prompt(
        claim_object=claim.claim_object,
        user_claim=claim.user_claim,
        submitted_image_ids=submitted_ids,
        evidence_requirements=evidence_reqs,
    )

    # Step 1-3: primary pass + validation circuit breaker.
    primary = _get_validated_assessment(
        primary_client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        images=images,
        temperature=PRIMARY_TEMPERATURE,
        model=PRIMARY_MODEL,
        ctx=ctx,
        trace=trace,
    )
    if primary is None:
        trace.hard_fallback = True
        trace.hard_fallback_reason = "validation failed twice"
        fallback = _hard_fallback_row(claim, "validation failed twice")
        return _finalize(claim, fallback, reference_data, escalation_triggered=False), trace

    trace.primary_confidence = primary.confidence

    # Step 4: confidence triage.
    if primary.confidence < CONFIDENCE_FLOOR:
        trace.hard_fallback = True
        trace.hard_fallback_reason = "primary confidence below floor"
        fallback = _hard_fallback_row(claim, "model self-reported low confidence")
        return _finalize(claim, fallback, reference_data, escalation_triggered=True), trace

    is_borderline = enable_self_consistency and (
        primary.confidence < CONFIDENCE_CEILING
        or bool(TRUST_RELEVANT_VISUAL_FLAGS & set(primary.visual_risk_flags))
    )
    if not is_borderline:
        return _finalize(claim, primary.model_dump(), reference_data, escalation_triggered=False), trace

    # Step 5: self-consistency second pass.
    trace.ran_self_consistency = True
    second = _get_validated_assessment(
        primary_client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        images=images,
        temperature=SELF_CONSISTENCY_TEMPERATURE,
        model=PRIMARY_MODEL,
        ctx=ctx,
        trace=trace,
    )
    if second is None:
        # Second pass itself failed validation twice -- don't silently keep
        # the unconfirmed first pass; route to manual review.
        trace.hard_fallback = True
        trace.hard_fallback_reason = "self-consistency pass failed validation twice"
        fallback = _hard_fallback_row(claim, "self-consistency pass failed validation")
        return _finalize(claim, fallback, reference_data, escalation_triggered=True), trace

    agree = (primary.claim_status == second.claim_status) and (primary.issue_type == second.issue_type)
    trace.self_consistency_agreed = agree
    if agree:
        return _finalize(claim, primary.model_dump(), reference_data, escalation_triggered=False), trace

    # Step 6: escalation tie-break -- only reached on actual disagreement.
    trace.ran_escalation = True
    escalation_prompt = build_escalation_prompt(
        claim_object=claim.claim_object,
        user_claim=claim.user_claim,
        submitted_image_ids=submitted_ids,
        pass_a_summary=_summarize(primary),
        pass_b_summary=_summarize(second),
    )
    tie_break = _get_validated_assessment(
        escalation_client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=escalation_prompt,
        images=images,
        temperature=0.0,
        model=ESCALATION_MODEL,
        ctx=ctx,
        trace=trace,
    )
    if tie_break is None or tie_break.confidence < CONFIDENCE_CEILING:
        trace.hard_fallback = True
        trace.hard_fallback_reason = "escalation could not confidently resolve disagreement"
        fallback = _hard_fallback_row(claim, "two independent reviews disagreed and the tie-break was not confident")
        return _finalize(claim, fallback, reference_data, escalation_triggered=True), trace

    return _finalize(claim, tie_break.model_dump(), reference_data, escalation_triggered=True), trace


def _finalize(
    claim: ClaimInput,
    assessment_dict: dict,
    reference_data: ReferenceData,
    *,
    escalation_triggered: bool,
) -> OutputRow:
    """Joins user_history (additive-only) and shapes the final output row.
    This is the ONLY function in the pipeline that touches user_history."""
    history = reference_data.history_for(claim.user_id)

    # merge_risk_flags only reads .valid_image and .visual_risk_flags (duck
    # typing) -- a SimpleNamespace lets hard-fallback dicts (which were
    # never validated as a real PrimaryAssessment) flow through the same
    # merge function as normal assessments, with no special-casing.
    stub = SimpleNamespace(
        valid_image=assessment_dict["valid_image"],
        visual_risk_flags=assessment_dict.get("visual_risk_flags", []),
    )
    risk_flags = merge_risk_flags(stub, history, escalated_to_manual_review=escalation_triggered)

    return OutputRow(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=assessment_dict["evidence_standard_met"],
        evidence_standard_met_reason=assessment_dict["evidence_standard_met_reason"],
        risk_flags=risk_flags,
        issue_type=assessment_dict["issue_type"],
        object_part=assessment_dict["object_part"],
        claim_status=assessment_dict["claim_status"],
        claim_status_justification=assessment_dict["claim_status_justification"],
        supporting_image_ids=assessment_dict["supporting_image_ids"],
        valid_image=assessment_dict["valid_image"],
        severity=assessment_dict["severity"],
    )
