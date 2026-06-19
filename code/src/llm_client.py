"""
LLM client abstraction.

Two implementations behind the same interface:
  - OpenAIClient: real calls via the OpenAI Responses API with native
    Structured Outputs (strict JSON schema). This is what produces the
    actual submission output.csv -- it requires OPENAI_API_KEY and network
    access to api.openai.com, which this build environment does not have.
  - MockLLMClient: deterministic, schema-valid stub responses. Used to
    prove the rest of the pipeline (validation, circuit breaker, borderline
    detection, self-consistency comparison, risk merge, CSV writing) is
    correct end-to-end without ever calling a network. This is how
    evaluation/dry_run_sample.csv and evaluation/dry_run_test.csv in this
    repo were produced.

Swapping between them is a single constructor call in pipeline.py / main.py
-- no code elsewhere needs to know which one is active.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.schemas import PrimaryAssessment

PRIMARY_MODEL = os.environ.get("CLAIM_REVIEW_PRIMARY_MODEL", "gpt-5.4-mini")
ESCALATION_MODEL = os.environ.get("CLAIM_REVIEW_ESCALATION_MODEL", "gpt-5.4")


class LLMClient(ABC):
    @abstractmethod
    def get_assessment(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[dict],
        temperature: float,
        model: str,
        retry_context: str | None = None,
    ) -> dict[str, Any]:
        """Returns a raw dict matching the PrimaryAssessment JSON shape.

        `retry_context`, when set, is the exact validation error from a
        prior attempt -- injected as an additional instruction so the
        retry is corrective, not a blind re-roll. See pipeline.py step 3.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------

def _strict_schema(model_cls: type) -> dict:
    """Pydantic's model_json_schema() output needs two adjustments for
    OpenAI's strict Structured Outputs mode: every object needs
    additionalProperties=False, and (in strict mode) every property must be
    listed as required -- optional fields are expressed via nullable types
    instead of omission. We only have one field with a default
    (image_observations' nested quality_or_trust_notes list default), and
    defaults are fine; strict mode cares about presence-in-"required", not
    about default values, so we just need the additionalProperties pass.
    """
    schema = model_cls.model_json_schema()

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(schema)
    return schema


class OpenAIClient(LLMClient):
    """Real OpenAI Responses API calls. Not exercised in this sandbox
    (api.openai.com is outside the sandbox's network allowlist) -- run this
    locally / in Claude Code with OPENAI_API_KEY set.

    NOTE ON API SURFACE: the Responses API's exact structured-output
    parameter shape has moved fast across 2025-2026 SDK releases. The shape
    below matches the Responses API documentation as of this writing
    (`text={"format": {"type": "json_schema", ...}}`, content blocks of
    type `input_text` / `input_image`). If your installed `openai` package
    version differs, check `client.responses.create.__doc__` or the current
    docs and adjust this one method -- the rest of the pipeline does not
    care how this method is implemented internally, only that it returns a
    dict matching PrimaryAssessment.

    Every call appends one line to `call_log_path` (default: call_log.jsonl
    in the cwd) with the model, real prompt/completion token counts from
    the API's own `usage` field, and wall-clock latency. The operational
    analysis in evaluation/evaluation_report.md is written from *estimates*
    because this sandbox can't make live calls -- the moment this runs for
    real, that log file replaces every estimate with measured numbers.
    """

    def __init__(self, api_key: str | None = None, call_log_path: str | Path = "call_log.jsonl"):
        from openai import OpenAI  # imported lazily so mock-mode runs don't need the package installed

        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._schema = _strict_schema(PrimaryAssessment)
        self._call_log_path = Path(call_log_path)

    def get_assessment(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[dict],
        temperature: float,
        model: str,
        retry_context: str | None = None,
    ) -> dict[str, Any]:
        content: list[dict] = [{"type": "input_text", "text": user_prompt}]
        for img in images:
            content.append({"type": "input_image", "image_url": img["data_url"]})

        if retry_context:
            content.append({
                "type": "input_text",
                "text": (
                    "Your previous response failed validation with this "
                    f"exact error, fix it and respond again with the full "
                    f"corrected JSON:\n{retry_context}"
                ),
            })

        # Transient-error retry (429 rate limit / 5xx) -- separate concern
        # from the schema-validation retry in pipeline.py, which retries
        # because the MODEL's output was wrong. This retries because the
        # CONNECTION was unreliable; the prompt is unchanged between
        # attempts. Capped at 3 attempts with exponential backoff + jitter,
        # which is the standard pattern and is enough headroom for a batch
        # job this size without risking a runaway retry storm.
        start = time.monotonic()
        response = self._call_with_backoff(model=model, temperature=temperature, content=content, system_prompt=system_prompt)
        latency_s = time.monotonic() - start

        self._log_call(model=model, response=response, latency_s=latency_s, n_images=len(images))
        return json.loads(response.output_text)

    def _call_with_backoff(self, *, model: str, temperature: float, content: list[dict], system_prompt: str, max_attempts: int = 3):
        import random as _random

        from openai import APIStatusError

        for attempt in range(max_attempts):
            try:
                return self._client.responses.create(
                    model=model,
                    temperature=temperature,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "primary_assessment",
                            "schema": self._schema,
                            "strict": True,
                        }
                    },
                )
            except APIStatusError as e:
                is_retryable = e.status_code == 429 or e.status_code >= 500
                if not is_retryable or attempt == max_attempts - 1:
                    raise
                sleep_s = (2 ** attempt) + _random.uniform(0, 0.5)
                time.sleep(sleep_s)

    def _log_call(self, *, model: str, response: Any, latency_s: float, n_images: int) -> None:
        usage = getattr(response, "usage", None)
        record = {
            "ts": time.time(),
            "model": model,
            "latency_s": round(latency_s, 3),
            "n_images": n_images,
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "cached_input_tokens": getattr(
                getattr(usage, "input_tokens_details", None), "cached_tokens", None
            ) if usage else None,
        }
        try:
            with self._call_log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # logging must never break the actual pipeline run


# ---------------------------------------------------------------------------
# Mock client -- offline, deterministic, schema-valid
# ---------------------------------------------------------------------------

_ISSUE_KEYWORDS = {
    "dent": "dent", "scratch": "scratch", "crack": "crack", "shatter": "glass_shatter",
    "broken": "broken_part", "missing": "missing_part", "torn": "torn_packaging",
    "crush": "crushed_packaging", "water": "water_damage", "stain": "stain",
}

_PART_KEYWORDS = {
    "bumper": "front_bumper", "rear": "rear_bumper", "door": "door", "hood": "hood",
    "windshield": "windshield", "mirror": "side_mirror", "headlight": "headlight",
    "taillight": "taillight", "fender": "fender", "screen": "screen",
    "keyboard": "keyboard", "trackpad": "trackpad", "hinge": "hinge", "lid": "lid",
    "box": "box", "seal": "seal", "label": "label", "content": "contents",
}


class MockLLMClient(LLMClient):
    """No network calls. Used by scripts/dry_run.py to prove the harness
    works (CSV parsing -> prompt building -> schema validation -> circuit
    breaker -> risk merge -> CSV writing) without an API key.

    Deliberately simple keyword matching -- this is NOT a model and isn't
    meant to approximate one. It exists purely so every non-LLM line of the
    pipeline can be exercised and unit-tested today.

    Set `force_low_confidence=True` to exercise the self-consistency /
    escalation path on demand (used in tests/test_pipeline.py).
    """

    def __init__(self, force_low_confidence: bool = False, force_disagree: bool = False):
        self.force_low_confidence = force_low_confidence
        self.force_disagree = force_disagree

    def get_assessment(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[dict],
        temperature: float,
        model: str,
        retry_context: str | None = None,
    ) -> dict[str, Any]:
        claim_text_lower = user_prompt.lower()
        image_ids = [img["image_id"] for img in images]

        issue_type = "unknown"
        for kw, val in _ISSUE_KEYWORDS.items():
            if kw in claim_text_lower:
                issue_type = val
                break

        object_part = "unknown"
        for kw, val in _PART_KEYWORDS.items():
            if kw in claim_text_lower:
                object_part = val
                break

        # Stable pseudo-randomness keyed on the prompt text so repeated
        # calls with identical input are reproducible, but a second,
        # independent self-consistency call (different temperature passed
        # in) can be made to differ via force_disagree.
        digest = hashlib.sha256(user_prompt.encode()).hexdigest()
        base_confidence = 0.55 + (int(digest[:2], 16) / 255) * 0.4  # 0.55-0.95

        confidence = 0.3 if self.force_low_confidence else base_confidence
        claim_status = "supported" if issue_type != "unknown" else "not_enough_information"
        if self.force_disagree:
            claim_status = "contradicted" if claim_status == "supported" else "supported"
            if issue_type == "unknown":
                issue_type = "dent"
                object_part = object_part if object_part != "unknown" else "body"

        supporting_ids = image_ids[:1] if claim_status != "not_enough_information" else []
        severity = "none" if issue_type == "none" else ("unknown" if issue_type == "unknown" else "medium")

        return {
            "image_observations": [
                {
                    "image_id": iid,
                    "object_type_matches_claim": True,
                    "relevant_part_visible": issue_type != "unknown",
                    "observed_condition": f"[mock] stub observation for {iid}",
                    "quality_or_trust_notes": [],
                }
                for iid in image_ids
            ],
            "evidence_standard_met": issue_type != "unknown",
            "evidence_standard_met_reason": "[mock] stub reason",
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": "[mock] stub justification",
            "supporting_image_ids": supporting_ids,
            "visual_risk_flags": [] if issue_type != "unknown" else ["damage_not_visible"],
            "valid_image": True,
            "severity": severity,
            "confidence": round(confidence, 2),
        }
