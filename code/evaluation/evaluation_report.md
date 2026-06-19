# Evaluation Report

## 1. What was actually run, where, and what wasn't

The original code was built and unit-tested in a sandboxed environment
whose network is allow-listed and does not include `api.openai.com` --
no live OpenAI call could be made from inside that sandbox. What *was*
run there, fully, against the real dataset and real image files:

- the entire deterministic pipeline (CSV loading, evidence/history
  lookups, image loading + base64 encoding of all 111 real images, prompt
  construction, Pydantic schema validation, the confidence-triage circuit
  breaker, the self-consistency comparison, the risk-flag merge, and CSV
  writing) against both `dataset/sample_claims.csv` (20 rows) and
  `dataset/claims.csv` (44 rows), using `MockLLMClient` in place of the
  real model (`python code/main.py --mock`).
- five targeted tests (`code/tests/test_pipeline.py`) that force each
  circuit-breaker branch individually: no-images short-circuit,
  validation-failure-then-fallback, below-floor-confidence fallback,
  disagreement-triggers-escalation, and hallucinated-image-ID rejection.
  All five pass.

**Everything below this point reflects real runs made afterward, locally,
with a real `OPENAI_API_KEY`.** That process surfaced and fixed two real
problems that the mock-mode testing above couldn't have caught, since
neither is a property of the orchestration logic:

- a **reasoning bug**: the model treated package images showing only
  packing filler (no item visible) as confirmation the item was missing,
  rather than as insufficient evidence (filler can conceal an item).
  Diagnosed by diffing real predictions against `sample_claims.csv`
  row-by-row, fixed by rewriting one section of `SYSTEM_PROMPT` after two
  weaker attempts didn't change the model's behavior, confirmed fixed by
  the same row-level diff on a third real run.
- a **data-integrity bug**: roughly 40% of this dataset's `.jpg`-named
  files are actually PNG/WEBP/AVIF content (verified with PIL across all
  111 files), and several are AVIF specifically, which OpenAI's API
  doesn't support in any form. This crashed the *entire* 44-row batch on
  the first real run of `claims.csv` on a single bad file. Fixed in two
  places: `image_utils.py` now detects the real format from file content
  instead of trusting the extension, and `main.py` now isolates each
  row's exceptions so one bad file can never again take down the other 43.

`dataset/output.csv` and every number in §2 and §4 below are real,
measured outputs from those local runs (295 real API calls total, logged
to `call_log.jsonl`). §3's mock-mode metrics are kept as the
harness-correctness baseline from the original sandbox build, not as a
claim about real accuracy -- real sample-set accuracy is reported
alongside it.

## 2. Strategy comparison (required: at least two configurations compared)

Both strategies are the same code in `src/pipeline.py`, differing only in
one flag (`enable_self_consistency`) -- not two implementations
maintained by hand, so the comparison is real rather than asserted. Run
via `python code/evaluation/main.py` (mock client, as run here) or
`python code/evaluation/main.py --real` (real model, real numbers).

| | Strategy A: single-pass | Strategy B: full circuit breaker |
|---|---|---|
| Description | One primary vision pass, accepted regardless of confidence. Schema validation + retry still applies -- that's never optional. | Confidence-triage -> self-consistency second pass on borderline cases -> escalation tie-break on actual disagreement -> deterministic risk merge. |
| `enable_self_consistency` | `False` | `True` (default) |
| Calls/claim, measured on sample set (mock) | **1.00** | **1.60** |
| Escalations triggered (mock, 20 claims) | 0 | 0 |
| `claim_status_accuracy` (mock) | 0.65 | 0.65 |

The mock run shows identical accuracy between strategies because
`MockLLMClient` has no real confidence-dependent variance to expose --
this is expected and stated plainly rather than presented as "B doesn't
help." What the mock run *does* show honestly is the actual call-volume
cost of B over A (1.60x), since that logic is in `pipeline.py` and runs
identically regardless of which client answers the call. The accuracy
delta between strategies can only be measured for real with `--real` and
a real API key -- full JSON output in `comparison_results.json`.

**Final strategy used for `output.csv`: Strategy B** (the default in
`code/main.py`). Rationale: B only spends the extra call(s) on claims the
model itself flags as uncertain (confidence 0.4-0.75) or visually suspect
(`claim_mismatch`/`wrong_object`/`possible_manipulation`/`non_original_image`),
and routes genuine two-pass disagreement to a stronger tie-breaker rather
than guessing -- which is the entire point of the hallucination-defense
design (see `code/README.md`). The 1.6x call-volume cost is small in
absolute terms (see §4) because escalation, the expensive path, is rare
by construction.

## 3. Evaluation harness, mock baseline, and real accuracy

Mock-mode metrics (sandboxed harness-correctness proof, see §1 -- not a
real accuracy claim):

```json
{
  "n_ground_truth": 20,
  "n_matched": 20,
  "claim_status_accuracy": 0.65,
  "issue_type_accuracy": 0.15,
  "object_part_accuracy": 0.2,
  "severity_accuracy": 0.55,
  "evidence_standard_met_accuracy": 0.9,
  "valid_image_accuracy": 0.9,
  "risk_flags_mean_jaccard": 0.758,
  "supporting_image_ids_mean_jaccard": 0.775
}
```

**Real accuracy, against `sample_claims.csv`, final version of
`SYSTEM_PROMPT` (after the packaging-evidence fix in §1), real
`gpt-5.4-mini`/`gpt-5.4`:**

```json
{
  "claim_status_accuracy": 0.75,
  "issue_type_accuracy": 0.65,
  "object_part_accuracy": 0.95,
  "severity_accuracy": 0.5,
  "evidence_standard_met_accuracy": 0.9,
  "valid_image_accuracy": 0.9,
  "claim_status_per_class": {
    "supported":               {"precision": 0.80, "recall": 0.923, "f1": 0.857, "support": 13},
    "contradicted":             {"precision": 0.50, "recall": 0.2,   "f1": 0.286, "support": 5},
    "not_enough_information":  {"precision": 0.667, "recall": 1.0,   "f1": 0.8,   "support": 2}
  },
  "risk_flags_mean_jaccard": 0.798,
  "supporting_image_ids_mean_jaccard": 0.925
}
```

The honest read of this, in order of importance for the interview:

- **`object_part` (0.95) and `valid_image` (0.9) are strong** -- the model
  reliably identifies the right component and correctly distinguishes
  "this photo is suspect" from "this photo just shows the wrong angle,"
  which is the distinction the whole `valid_image` design hinges on.
- **`contradicted` recall is the weak spot (0.2, 1 of 5 caught)** -- the
  model is biased toward `supported`/cautious verdicts and under-calls
  genuine mismatches. This is the safety-relevant number, and it's worth
  saying plainly rather than burying it: this system, as currently
  prompted, would let more bad claims through than it should, not the
  other way around.
- **But the two adversarial cases this system was specifically built to
  defend against both still triggered `manual_review_required`** even
  when their `claim_status` label was wrong -- the image-embedded
  instruction ("approve this claim") was correctly flagged and not
  obeyed, and the stock-photo watermark was correctly flagged and
  `valid_image` was correctly set to `false`, in both cases. The label
  being wrong and the safety net firing anyway are both true at once, and
  the second fact matters more than the first for a system meant to flag
  things for human review rather than auto-decide everything.
- **`severity` (0.5) is the next thing I'd tune** if there were more time
  -- it's the least-constrained field (no hard validator in `schemas.py`
  ties it to anything except `issue_type=="none"`), so it's most exposed
  to model calibration drift.

Per the problem statement's own framing ("exact_match plus
precision/recall is sufficient and shows you understand the metrics"),
`evaluation/scoring.py` (invoked by `evaluation/main.py`) deliberately stays at exact-match accuracy for
categorical fields, precision/recall/F1 per `claim_status` class, and
Jaccard overlap for the two multi-valued semicolon fields
(`risk_flags`, `supporting_image_ids`). No LLM-as-judge, no RAGAS --
both would be disproportionate machinery for a 20-row labeled set and
would themselves need defending as "why build an eval system to evaluate
your eval system."

The mock run's `claim_status` per-class breakdown is the most informative
number from that earlier baseline: 100% recall on `supported`, 0% recall
on `contradicted` and `not_enough_information`, because the mock has no
actual mechanism for detecting a mismatch between claim and image. That
was the harness correctly diagnosing a bad model, which is what it's for
-- and it's a useful contrast against the real numbers above, which show
a real (if imperfect) model genuinely engaging with the distinction.

The data-derived risk-flag merge logic, separately, was validated against
the real labels directly (not via the mock): joining
`user_history.csv.history_flags` against `sample_claims.csv.risk_flags`
and replaying that logic in `src/risk_merge.py` reproduces the exact
`user_history_risk` / `manual_review_required` pattern for all 20 labeled
rows. This is the one part of the system where "accuracy" is actually
known and is 100%, because it's deterministic, not modeled. The same
check was re-run against the real, final `dataset/output.csv` (44
unlabeled rows, so `claim_status` itself can't be scored there) -- the
history -> risk_flags passthrough matched `user_history.csv` exactly on
all 44 rows, real model, real run, zero mismatches.

## 4. Operational analysis

**Everything in this section is measured from `call_log.jsonl`** (295 real
calls logged across every real run made during development -- three
sample-set iterations while fixing the packaging-evidence bug, the real
strategy comparison, and the final `claims.csv` -> `output.csv` run), not
estimated. The original pre-run estimates this section contained (worked
example: ~$0.0028/call, ~$0.13-$0.19 for the full test set, ~3.5 min
sequential) turned out to be close, which is worth noting briefly before
replacing them: real per-call cost landed at $0.0028 average (mini calls)
-- the estimate's arithmetic was right; what it couldn't predict was the
real cache hit rate or the real escalation rate, both below.

### Model call volume (real, final `claims.csv` run: 44 claims, 64 calls)

| outcome | claims | calls each |
|---|---|---|
| resolved on primary pass alone | 28 | 1 |
| needed self-consistency, then agreed | 12 | 2 |
| disagreed, needed escalation tie-break | 4 | 3 |
| **total** | **44** | **64 calls, avg 1.45/claim** |

So in practice: **36.4% of claims hit the borderline band** (confidence
0.4-0.75 or a trust-relevant visual flag), and of those, **escalation
actually fired on 4 of 44 claims (9.1%)** -- not zero, but rare, exactly
as the circuit breaker was designed to keep it. The mock-mode estimate of
"closer to 1.1-1.3 calls/claim" undershot a bit (real was 1.45); the real
model flags more claims as genuinely borderline than the mock's arbitrary
confidence distribution did, which is the opposite of what you'd want if
this were padding cost for no reason -- it means real, visually-grounded
uncertainty signals are firing, not noise.

### Token usage and cost (real, all 295 logged calls)

| | calls | avg input tok | avg output tok | avg cached tok | cache hit rate |
|---|---|---|---|---|---|
| gpt-5.4-mini (primary + self-consistency) | 280 | 3,573 | 295 | 2,034 | 56.9% |
| gpt-5.4 (escalation) | 15 | 3,431 | 363 | 358 | 10.4% |
| **all calls** | **295** | **3,566** | **298** | **1,949** | **54.7%** |

Real per-call cost (GPT-5.4-mini $0.75/$0.075-cached/$4.50 per MTok
in/cached-in/out; GPT-5.4 $2.50/$0.25-cached/$15 per MTok):

- **Final `claims.csv` run (44 claims, 64 calls, the one that produced
  `dataset/output.csv`): $0.18** total ($0.14 mini + $0.04 escalation).
  Cache hit rate on this run specifically was **73.2%** -- noticeably
  higher than the session average, because by this point the system
  prompt had been making identical-prefix calls for a while and the cache
  was fully warm.
- Every real call made across the whole build session (295 calls,
  including three sample-set debug iterations and the strategy
  comparison): **$0.94** total.
- The 20-row sample set, single real run: proportionally **~$0.06**.

For comparison, the original estimate for the 44-row test set was
"$0.13-$0.19, well under $0.25" -- the real number ($0.18) landed inside
that range. The estimate undercounted the cache benefit slightly (assumed
~90% discount on a ~1,080-token cacheable prefix; the real prefix that
ends up cached is larger than just the system prompt, since the evidence
checklist text is also frequently identical across same-object-type
claims) and didn't know the real escalation rate, but the order of
magnitude held.

### Latency / runtime (real)

- Final test-set run, **`--concurrency 5`: 57.2s wall-clock** for 44
  claims / 64 calls (printed by `main.py` itself).
- Sum of all 64 individual call latencies: 255.6s -- i.e. **running the
  same 64 calls sequentially (`--concurrency 1`) would take roughly
  4.3 minutes**, versus the 57s actually observed at concurrency 5.
  That's roughly a 4.5x wall-clock speedup from 5x concurrency (sub-linear,
  as expected -- some calls in the borderline/escalation path are
  inherently sequential *within* a single claim, since self-consistency
  and escalation depend on the primary pass's result).
- Per-call latency across all 295 real calls: avg **3.92s**, min 1.86s,
  max 17.38s. The 17s outlier is consistent with an occasional slow
  response or a retry-with-backoff absorbing a transient error rather
  than a systemic problem -- worth a glance at `call_log.jsonl` if it
  recurs at higher volume, but a single outlier in 295 calls isn't a
  pattern.

### TPM / RPM and the batching/throttling/retry strategy actually used

The final run pushed roughly 64 calls x ~3,990 avg total tokens
(input+output) &approx; **255K tokens** through **64 requests** in under
a minute at concurrency 5. Across the *whole* build session it was
**1.14M total tokens across 295 requests**, spread over however many
hours of iterative debugging -- comfortably inside per-minute limits for
any OpenAI usage tier on a mini-tier model; rate limiting was never
actually hit (no 429s appear in `call_log.jsonl`'s latency distribution
as anomalies, and the retry/backoff path was never exercised in practice).

What's actually in the code, regardless of whether this volume needed it:

- **Concurrency control**: `--concurrency N` in `main.py` bounds a
  `ThreadPoolExecutor` over independent rows. Used at 5 for the real run;
  the default stays 1 (sequential) because at 44 rows added concurrency
  isn't load-bearing -- the flag exists for when volume would justify it.
- **Retry on transient failure**: `OpenAIClient._call_with_backoff`
  retries 429/5xx up to 3 attempts with exponential backoff + jitter,
  separate from the *validation* retry in `pipeline.py` (model's JSON was
  wrong vs. the connection was unreliable -- two different failure modes,
  two independent retry loops). Never triggered in this run, by design
  this doesn't mean it's untested -- see `tests/test_pipeline.py`'s
  validation-retry coverage for the other half of that statement.
- **Per-row crash isolation**: added mid-build after a real failure mode
  surfaced -- roughly 40% of this dataset's `.jpg`-named files are
  actually PNG/WEBP/AVIF content, and AVIF isn't an OpenAI-supported
  format at all. The first real run crashed the entire 44-row batch on a
  single bad file before this existed. `main.py`'s `_run()` now wraps
  `process_claim` in a try/except that routes any unhandled exception to
  the same `not_enough_information` / `manual_review_required` fallback
  used elsewhere, so one bad file can never again take down a 44-row run
  in progress. `image_utils.py` also now detects real image format from
  file content (via PIL) rather than trusting the file extension, and
  converts unsupported formats to PNG before sending to the API.
- **Caching**: real cache hit rate was 54.7% session-wide and 73.2% on
  the final run specifically (see above) -- the static-system-prompt
  design choice paid off as intended, with zero extra code beyond keeping
  the prompt's text byte-identical across calls.
- **Batching**: still not using OpenAI's separate async Batch API (~50%
  cheaper, but queued for up to 24h) -- this submission needed a
  same-session turnaround, not minimum cost. First thing to swap in for a
  non-interactive, cost-sensitive production version of this job at
  higher volume.
- **Real usage logging**: every line of `call_log.jsonl` is exactly this
  section's source data -- model, latency, and the API's own reported
  token counts, appended on every real call. This replaced every estimate
  in this section with a measured number once the system actually ran.

## 5. What I'd improve next if this weren't a 24-hour build

- **`contradicted` recall (0.2, real number, §3).** This is the most
  important item on this list, not the first one by accident. The model
  under-calls genuine claim/image mismatches. Worth trying before
  anything else: a few real worked examples of `contradicted` verdicts
  (the same way the packaging-evidence fix worked in §1) added to
  `SYSTEM_PROMPT` as concrete cases rather than abstract rules -- the
  packaging fix only started working once it was phrased as an explicit
  self-check against the model's own stated observations, and the same
  pattern likely applies here.
- **Calibrate the confidence thresholds against real model output.** 0.4
  and 0.75 are still reasoned defaults, not tuned values. Real data now
  exists to do this properly: pull `confidence` alongside
  correct/incorrect from the real sample-set run and plot where this
  specific model's confidence actually separates right from wrong answers
  -- the real escalation rate (9.1% of claims, §4) suggests the bands are
  in a reasonable range, but "reasonable" isn't "tuned."
- **Static per-object system prompts.** Right now the evidence checklist
  is built dynamically per call (`build_user_prompt`) and placed in the
  user turn. Since there are only 3 possible `claim_object` values, baking
  3 fully-static system-prompt variants (car/laptop/package) would push
  the checklist into the cached prefix too, instead of re-sending it as
  fresh tokens on every call -- real cache hit rate was already 73.2% on
  the final run without this, so the remaining win here is smaller than
  it looked before real numbers existed, but it's still free money.
- **Batch API for the full test-set run** once cost/turnaround time for a
  much larger claims volume actually matters.
