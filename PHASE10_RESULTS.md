# Phase 10 Evaluation Results

Baseline agent (bare Qwen2.5-7B-Instruct, 4-bit, single-shot generation) vs.
controlled agent (full Phase 1-9 pipeline: LangGraph + RiskState + adaptive
router + responsibility layer + budget + policy engine + recovery loop),
evaluated on n=100 HaluEval `dialogue_data.json` test-split questions, real
generation (not prefill-only), single seed (42).

Test set was touched three times during Phase 10 development (n=3 smoke,
n=20, n=100) due to real infrastructure bugs requiring fixes between runs
(missing package on Kaggle, wrong file path, injection classifier
miscalibration) -- not tuning of thresholds, rules, or model parameters. This
deviates from "touched once" and is recorded here rather than glossed over.
The n=100 run is the single authoritative result reported below.

## Table 1: Primary result -- paired difference on knowledge-clean subset (n=82)

Metric: NLI contradiction score of the agent's response against the
example's `knowledge` field (lower = more factually grounded). Paired per
example (same question, both arms) with a 10,000-resample bootstrap 95% CI
on the mean difference.

| | value |
|---|---|
| baseline mean contradiction | 0.616 (95% CI [0.536, 0.696]) |
| controlled mean contradiction | 0.429 (95% CI [0.345, 0.514]) |
| **paired difference (baseline - controlled)** | **+0.187, 95% CI [+0.113, +0.266]** |

CI excludes 0 -- the improvement is statistically significant. Pairing (same
example, both arms) rather than comparing independent group means gives most
of this statistical power; a single seed was judged sufficient since seeds
measure generation-sampling variance, which pairing does not address, but
example-to-example variance (the dominant source at this scale) is exactly
what pairing controls for.

## Table 2: Secondary -- all 100 examples (including 18 bad-knowledge)

| | value |
|---|---|
| paired difference (baseline - controlled) | +0.149, 95% CI [+0.087, +0.214] |

Still significant, smaller effect size, as expected -- bad-knowledge examples
add noise to a metric that is only as good as the knowledge field it compares
against.

## Table 3: Disaggregated recovery-loop outcomes (n=60 interventions, knowledge-clean subset)

Of the 82 knowledge-clean examples, 22 needed no intervention (cheap checks
passed immediately) and 60 triggered the recovery loop. Outcome of those 60,
each with a Wilson 95% CI:

| outcome | n | rate | 95% CI |
|---|---|---|---|
| Succeeded (reached ALLOW) | 27/60 | 45.0% | [33.1%, 57.5%] |
| Improved but still flagged (exhausted 3-round budget) | 11/60 | 18.3% | [10.6%, 29.9%] |
| No meaningful change (\|delta\| <= 0.05) | 12/60 | 20.0% | [11.8%, 31.8%] |
| **Made worse** | **10/60** | **16.7%** | **[9.3%, 28.0%]** |

This is reported as a first-class result, not a footnote under the mean:
recovery is not a universal fix. Nearly half of interventions succeed
outright, but a substantial minority -- genuinely somewhere between 1-in-11
and 1-in-4 by the confidence interval -- make the response worse, at real
cost (each intervention burns up to 4 real generation calls).

## Table 4: Residual injection classifier false-positive rate

Measured directly on all 100 real tool outputs (the `knowledge` field routed
through `retrieval_tool`, then scored by `injection_check`), not inferred
from policy actions.

| | value |
|---|---|
| False positives on benign tool output | 1/100 = 1.0%, Wilson 95% CI [0.2%, 5.4%] |

Down from an earlier-measured 62.5% (8-example set) for the originally-chosen
`deepset/deberta-v3-base-injection` classifier; the swap to
`protectai/deberta-v3-base-prompt-injection-v2` (verified to still catch
genuine injections at >0.999 confidence) resolved nearly all of it. The one
residual false positive (example 7, the Narnia/Pauline-Baynes example) is
also one of the confirmed-bad-knowledge examples excluded from Table 1's
primary metric -- a coincidence worth noting, not a confound between the two
issues (wrong author attribution vs. classifier miscalibration are unrelated
root causes).

## Knowledge-audit methodology

All 100 `knowledge` fields were manually graded by Claude checking HaluEval's
claims against its own training knowledge -- **this is an LLM's
self-assessment of another dataset's factual claims, not independently
verified ground truth.** A project about the unreliability of LLM factuality
judgments should not quietly use an LLM as its own arbiter of truth; this
audit inherits the same reliability limits this whole project exists to
detect, and is reported as such rather than treated as authoritative.

Confidence tiers used:

| tier | count | criterion |
|---|---|---|
| Confirmed bad | 11/100 | high confidence, clear factual error (e.g. a translator credited as author, a genre misclassification, a self-contradicting knowledge string) |
| Probable bad | 7/100 | moderate-good confidence, likely error but less certain |
| Uncertain | 4/100 | genuine doubt, not counted as bad either way |
| No issue found | 78/100 | -- |

Confirmed + probable (18/100, 18%) were excluded from Table 1's primary
metric. This rate, both at n=20 (15-20%) and n=100 (18%), is well above a
"few outliers" threshold and is the reason the vs-knowledge contradiction
score is not treated as a fully reliable ground-truth metric on its own (see
Limitations).

## Limitations

- **Single seed.** Seeds were deliberately not added -- they measure
  generation-sampling variance (temperature-driven differences in what the
  model generates on retry), which is a different source of uncertainty from
  the example-to-example variance that pairing already controls for at this
  n. The paired CI already excludes zero without them. Still, single-seed
  means any given regeneration's specific wording is not itself statistically
  characterized -- only the aggregate effect across 100 different questions
  is.
- **HaluEval cross-model provenance.** The dataset's original hallucinated/
  correct response pairs were generated by a different model than the one
  used here (Qwen2.5-7B-Instruct); Phase 10 doesn't use those pre-written
  responses at all (both agents generate fresh text), so this specific
  limitation is less load-bearing for Phase 10 than it was for Phase 3's
  probe training, but the underlying `knowledge` fields still come from
  HaluEval's original construction pipeline, which is where the 18%
  error rate above likely originates.
- **vs-knowledge metric reliability.** 18% of knowledge fields contain
  confirmed or probable factual errors (audited by an LLM, see above, itself
  a limitation). The metric is only as good as its ground truth; results are
  reported on the knowledge-clean subset as primary specifically to address
  this, but the audit itself is not independently verified.
- **n=100.** A single stratified draw from the test split. Real, but modest;
  larger n would tighten every CI above further.
- **Real Kaggle GPU cost.** This n=100 run used a real, measured 5:28:04 of
  the weekly 30-hour GPU quota (18.2%), confirmed via Kaggle's own quota API
  (`kaggle kernels quota_view`), not estimated. ~24.5 hours remained after
  this run.
