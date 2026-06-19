"""
CLI entrypoint.

Run from the repo root (dataset/ is a sibling of code/, not inside it):

    python code/main.py --input dataset/sample_claims.csv --output code/evaluation/dry_run_sample.csv --mock
    python code/main.py --input dataset/claims.csv --output dataset/output.csv --concurrency 5
    python code/main.py --input dataset/claims.csv --output dataset/output.csv --mock   # harness check, no API key needed

(Or `cd code && python main.py ...` -- --dataset-root defaults to ../dataset
relative to this file either way, so both invocation styles resolve to the
same place.)

Concurrency note: rows are independent (each is a separate claim with its
own images), so concurrency is a plain bounded ThreadPoolExecutor over
process_claim, not asyncio. The pipeline is I/O-bound on HTTP calls to the
OpenAI API; --concurrency is the one knob that controls how many of those
are in flight at once, which is what actually matters for staying under a
per-key rate limit. Default is 1 (sequential) because the test set here is
44 rows -- at that size, added concurrency complexity isn't earning its
keep; the flag exists for when it would.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import ReferenceData
from src.csv_writer import write_output_csv
from src.llm_client import LLMClient, MockLLMClient, OpenAIClient
from src.pipeline import ClaimInput, process_claim, build_error_fallback_row, PipelineTrace


def load_claims(path: Path) -> list[ClaimInput]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            ClaimInput(
                user_id=row["user_id"],
                image_paths=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
            )
            for row in reader
        ]


def build_clients(use_mock: bool) -> tuple[LLMClient, LLMClient]:
    if use_mock:
        return MockLLMClient(), MockLLMClient()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it, or pass --mock to run the "
            "offline harness check instead."
        )
    client = OpenAIClient(api_key=api_key)
    return client, client  # escalation reuses the same client; only the model string differs


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-modal evidence review pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Path to claims CSV (sample_claims.csv or claims.csv)")
    parser.add_argument("--output", required=True, type=Path, help="Path to write output CSV")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT.parent / "dataset", help="Root containing image_paths (default: ../dataset, i.e. the repo-level dataset/ folder)")
    parser.add_argument("--mock", action="store_true", help="Use the offline mock LLM client instead of real OpenAI calls")
    parser.add_argument("--concurrency", type=int, default=1, help="Max claims processed in parallel (default: 1, sequential)")
    parser.add_argument("--no-self-consistency", action="store_true", help="Disable the self-consistency/escalation circuit breaker (Strategy A, see evaluation/main.py) -- single pass only")
    args = parser.parse_args()

    reference_data = ReferenceData(
        evidence_requirements_csv=args.dataset_root / "evidence_requirements.csv",
        user_history_csv=args.dataset_root / "user_history.csv",
    )
    claims = load_claims(args.input)
    primary_client, escalation_client = build_clients(args.mock)

    print(f"Processing {len(claims)} claims (mock={args.mock}, concurrency={args.concurrency})...")
    start = time.time()

    results: list = [None] * len(claims)
    call_counts = []

    def _run(i: int, claim: ClaimInput):
        try:
            row, trace = process_claim(
                claim,
                dataset_root=args.dataset_root,
                reference_data=reference_data,
                primary_client=primary_client,
                escalation_client=escalation_client,
                enable_self_consistency=not args.no_self_consistency,
            )
        except Exception as e:
            print(f"  [ERROR] user={claim.user_id}: {type(e).__name__}: {e} -- falling back to manual review")
            row = build_error_fallback_row(claim, reference_data, f"{type(e).__name__}: {e}")
            trace = PipelineTrace(hard_fallback=True, hard_fallback_reason=str(e))
        return i, row, trace

    if args.concurrency <= 1:
        for i, claim in enumerate(claims):
            i, row, trace = _run(i, claim)
            results[i] = row
            call_counts.append(trace.model_calls)
            print(f"  [{i+1}/{len(claims)}] user={claim.user_id} -> {row.claim_status} (calls={trace.model_calls})")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_run, i, c) for i, c in enumerate(claims)]
            done = 0
            for fut in as_completed(futures):
                i, row, trace = fut.result()
                results[i] = row
                call_counts.append(trace.model_calls)
                done += 1
                print(f"  [{done}/{len(claims)}] user={claims[i].user_id} -> {row.claim_status} (calls={trace.model_calls})")

    elapsed = time.time() - start
    write_output_csv(results, args.output)
    print(f"\nWrote {len(results)} rows to {args.output}")
    print(f"Elapsed: {elapsed:.1f}s | total model calls: {sum(call_counts)} | avg calls/claim: {sum(call_counts)/len(call_counts):.2f}")


if __name__ == "__main__":
    main()
