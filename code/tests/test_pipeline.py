"""
Tests that exercise each branch of the pipeline's circuit breaker
explicitly. The mock dry-run scripts prove the happy path doesn't crash;
these prove the SAFETY paths actually fire, which is the part that matters
for the "how do you avoid guessing" question.

Run with: python -m pytest tests/ -v
(or: python tests/test_pipeline.py  -- also runs standalone, no pytest dependency)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent  # code/
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import ReferenceData
from src.llm_client import LLMClient, MockLLMClient
from src.pipeline import process_claim, ClaimInput
from src.schemas import PrimaryAssessment, validate_against_context, PrimaryAssessmentContext

DATASET_ROOT = REPO_ROOT.parent / "dataset"  # sibling of code/, at the actual repo root
REFERENCE_DATA = ReferenceData(
    evidence_requirements_csv=DATASET_ROOT / "evidence_requirements.csv",
    user_history_csv=DATASET_ROOT / "user_history.csv",
)

# A real claim from the sample set, used as a fixture across tests.
SAMPLE_CLAIM = ClaimInput(
    user_id="user_001",
    image_paths="images/sample/case_001/img_1.jpg",
    user_claim="Customer: There is a dent on the rear bumper.",
    claim_object="car",
)


class AlwaysInvalidClient(LLMClient):
    """Simulates a model that never produces schema-valid output -- tests
    the hard-fallback-after-retry path."""
    def get_assessment(self, **kwargs) -> dict[str, Any]:
        return {"this": "is not a valid PrimaryAssessment"}


class NoImagesClaimClient(LLMClient):
    def get_assessment(self, **kwargs) -> dict[str, Any]:
        raise AssertionError("Should never be called when there are no submitted images")


def test_no_images_short_circuits_without_calling_llm():
    claim = ClaimInput(user_id="user_001", image_paths="", user_claim="x", claim_object="car")
    row, trace = process_claim(
        claim, dataset_root=DATASET_ROOT, reference_data=REFERENCE_DATA,
        primary_client=NoImagesClaimClient(),
    )
    assert row.claim_status == "not_enough_information"
    assert trace.model_calls == 0
    print("PASS: no-images claim short-circuits without any LLM call")


def test_validation_failure_falls_back_after_one_retry():
    row, trace = process_claim(
        SAMPLE_CLAIM, dataset_root=DATASET_ROOT, reference_data=REFERENCE_DATA,
        primary_client=AlwaysInvalidClient(),
    )
    assert row.claim_status == "not_enough_information"
    assert row.valid_image is False
    assert "manual_review_required" in row.risk_flags
    assert trace.hard_fallback is True
    assert trace.model_calls == 2  # one attempt + one retry, then give up
    print("PASS: schema-invalid output retries once then hard-falls-back, never crashes")


def test_low_confidence_routes_to_fallback_without_second_call():
    row, trace = process_claim(
        SAMPLE_CLAIM, dataset_root=DATASET_ROOT, reference_data=REFERENCE_DATA,
        primary_client=MockLLMClient(force_low_confidence=True),
    )
    assert row.claim_status == "not_enough_information"
    assert trace.hard_fallback_reason == "primary confidence below floor"
    assert trace.ran_self_consistency is False  # below-floor skips straight to fallback
    assert trace.model_calls == 1
    print("PASS: below-floor confidence skips self-consistency and falls back in a single call")


def test_disagreement_triggers_escalation_and_resolves():
    primary = MockLLMClient(force_low_confidence=False)
    # force_disagree only affects the SECOND call in a real client; here we
    # simulate disagreement by giving the escalation client a confident,
    # valid response so the tie-break resolves cleanly.
    row, trace = process_claim(
        SAMPLE_CLAIM, dataset_root=DATASET_ROOT, reference_data=REFERENCE_DATA,
        primary_client=_DisagreeingMock(),
        escalation_client=MockLLMClient(),
    )
    assert trace.ran_self_consistency is True
    assert trace.self_consistency_agreed is False
    assert trace.ran_escalation is True
    print(f"PASS: disagreement correctly triggers escalation (resolved to {row.claim_status})")


class _DisagreeingMock(LLMClient):
    """Returns a low/borderline-confidence answer on the first call, and a
    contradicting answer on the second (self-consistency) call -- forces
    the escalation path deterministically for testing."""
    def __init__(self):
        self._calls = 0

    def get_assessment(self, **kwargs) -> dict[str, Any]:
        self._calls += 1
        base = MockLLMClient().get_assessment(**kwargs)
        base["confidence"] = 0.6  # inside the borderline band
        if self._calls == 1:
            base["claim_status"] = "supported"
            base["issue_type"] = "dent"
            base["object_part"] = "rear_bumper"
            base["supporting_image_ids"] = ["img_1"]
        else:
            base["claim_status"] = "contradicted"
            base["issue_type"] = "scratch"
            base["object_part"] = "rear_bumper"
            base["supporting_image_ids"] = ["img_1"]
        return base


def test_hallucinated_image_id_is_rejected():
    """A model that cites an image ID never submitted must fail validation,
    not be silently accepted."""
    ctx = PrimaryAssessmentContext(submitted_image_ids=["img_1"], claim_object="car")
    bad = PrimaryAssessment(
        image_observations=[{
            "image_id": "img_1", "object_type_matches_claim": True,
            "relevant_part_visible": True, "observed_condition": "x",
            "quality_or_trust_notes": [],
        }],
        evidence_standard_met=True, evidence_standard_met_reason="x",
        issue_type="dent", object_part="rear_bumper",
        claim_status="supported", claim_status_justification="x",
        supporting_image_ids=["img_1", "img_99"],  # img_99 was never submitted
        visual_risk_flags=[], valid_image=True, severity="medium", confidence=0.9,
    )
    errors = validate_against_context(bad, ctx)
    assert any("img_99" in e for e in errors)
    print("PASS: hallucinated image ID is caught by context validation")


if __name__ == "__main__":
    test_no_images_short_circuits_without_calling_llm()
    test_validation_failure_falls_back_after_one_retry()
    test_low_confidence_routes_to_fallback_without_second_call()
    test_disagreement_triggers_escalation_and_resolves()
    test_hallucinated_image_id_is_rejected()
    print("\nAll circuit-breaker tests passed.")
