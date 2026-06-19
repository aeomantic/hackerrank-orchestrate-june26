"""
Evaluation entrypoint (suggested per AGENTS.md / README.md).

Runs and compares two real, runnable configurations of the same pipeline
against dataset/sample_claims.csv, then states which one output.csv was
generated with and why:

  Strategy A: single primary vision pass, accepted regardless of
              confidence (enable_self_consistency=False). Still validated
              and retried on schema failure -- that circuit breaker is
              never optional, it's not part of what's being compared here.
  Strategy B: the full pipeline -- confidence-triage, self-consistency
              second pass on borderline cases, escalation tie-break on
              disagreement (enable_self_consistency=True, the default).

These are not two different implementations to keep in sync by hand; B is
A with one extra code path turned on (src/pipeline.py's
`enable_self_consistency` flag), so the comparison is real, not asserted.

Usage:
    python evaluation/main.py                 # mock client (no API key needed, what was actually run here)
    python evaluation/main.py --real          # real OpenAI calls (needs OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from src.data_loader import ReferenceData
from src.llm_client import LLMClient, MockLLMClient, OpenAIClient
from src.pipeline import ClaimInput, process_claim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import evaluate_rows, load_rows  # noqa: E402

DATASET_ROOT = CODE_ROOT.parent / "dataset"
SAMPLE_CSV = DATASET_ROOT / "sample_claims.csv"


def load_claims(path: Path) -> list[ClaimInput]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            ClaimInput(
                user_id=r["user_id"], image_paths=r["image_paths"],
                user_claim=r["user_claim"], claim_object=r["claim_object"],
            )
            for r in csv.DictReader(f)
        ]


def run_strategy(
    claims: list[ClaimInput],
    *,
    reference_data: ReferenceData,
    client: LLMClient,
    enable_self_consistency: bool,
) -> tuple[dict[tuple, dict], dict]:
    """Returns (predictions keyed like scoring.load_rows, summary stats)."""
    preds: dict[tuple, dict] = {}
    total_calls = 0
    escalations = 0
    start = time.time()

    for claim in claims:
        row, trace = process_claim(
            claim,
            dataset_root=DATASET_ROOT,
            reference_data=reference_data,
            primary_client=client,
            escalation_client=client,
            enable_self_consistency=enable_self_consistency,
        )
        d = row.model_dump()
        d["evidence_standard_met"] = str(d["evidence_standard_met"]).lower()
        d["valid_image"] = str(d["valid_image"]).lower()
        preds[(claim.user_id, claim.image_paths)] = d
        total_calls += trace.model_calls
        escalations += int(trace.ran_escalation)

    elapsed = time.time() - start
    summary = {
        "n_claims": len(claims),
        "total_model_calls": total_calls,
        "avg_calls_per_claim": round(total_calls / len(claims), 2) if claims else 0,
        "escalations_triggered": escalations,
        "elapsed_s": round(elapsed, 2),
    }
    return preds, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real OpenAI calls instead of the mock client (needs OPENAI_API_KEY)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "comparison_results.json")
    args = parser.parse_args()

    if args.real:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set -- omit --real to use the mock client instead.")
        client: LLMClient = OpenAIClient(api_key=api_key)
        mode = "real"
    else:
        client = MockLLMClient()
        mode = "mock"

    reference_data = ReferenceData(
        evidence_requirements_csv=DATASET_ROOT / "evidence_requirements.csv",
        user_history_csv=DATASET_ROOT / "user_history.csv",
    )
    claims = load_claims(SAMPLE_CSV)
    truth = load_rows(SAMPLE_CSV)

    print(f"Evaluating on {len(claims)} rows from dataset/sample_claims.csv (client={mode})\n")

    print("Running Strategy A (single-pass, no self-consistency)...")
    preds_a, summary_a = run_strategy(claims, reference_data=reference_data, client=client, enable_self_consistency=False)
    metrics_a = evaluate_rows(preds_a, truth)

    print("Running Strategy B (full circuit breaker: confidence-triage + self-consistency + escalation)...")
    preds_b, summary_b = run_strategy(claims, reference_data=reference_data, client=client, enable_self_consistency=True)
    metrics_b = evaluate_rows(preds_b, truth)

    result = {
        "mode": mode,
        "strategy_a_single_pass": {"summary": summary_a, "metrics": metrics_a},
        "strategy_b_full_circuit_breaker": {"summary": summary_b, "metrics": metrics_b},
        "final_strategy_used_for_output_csv": "strategy_b_full_circuit_breaker",
        "rationale": (
            "Strategy B is what main.py runs by default and what produced output.csv. "
            "Strategy A spends roughly one model call per claim and accepts the first answer "
            "regardless of confidence; B spends more calls only on claims the model itself "
            "flags as uncertain (confidence 0.4-0.75) or visually suspect (claim_mismatch / "
            "wrong_object / possible_manipulation / non_original_image), and routes genuine "
            "two-pass disagreement to a stronger tie-breaker model rather than guessing. The "
            "cost delta (see avg_calls_per_claim above) is small because escalation is rare by "
            "construction -- it only fires on actual disagreement, not on every borderline case."
        ),
        "caveat": (
            "Both strategies were run through MockLLMClient in this build environment "
            "(no network access to api.openai.com here -- see README). The mock is a "
            "keyword stub, not a vision model, so the *metrics* above mainly demonstrate "
            "that the comparison harness itself works, not which strategy is more accurate "
            "in reality. The *call-volume* numbers (avg_calls_per_claim, escalations_triggered) "
            "ARE structurally meaningful regardless of which client is behind LLMClient, since "
            "that logic lives in src/pipeline.py and doesn't change based on who answers. "
            "Re-run with --real and a real OPENAI_API_KEY to get real accuracy numbers."
        ) if mode == "mock" else None,
    }

    print("\n" + json.dumps(result, indent=2))
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
