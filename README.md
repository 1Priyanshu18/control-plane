# control_planeAI

A control layer around an LLM agent: observe the request, classify its risk,
run only the checks that risk profile requires, decide via a deterministic
policy engine, intervene (verify/retry/regenerate/redact/restrict/block), and
learn from what interventions actually did.

## Architecture

Five responsibilities, kept separate on purpose:

- **LangGraph** executes the agent's steps and checkpoints state at each node
  -- this is what lets a recovery loop roll back to before a failed step and
  re-run it, instead of restarting from scratch.
- **RiskState** holds seven risk dimensions (factuality, injection, PII,
  toxicity, self-consistency, tool risk, budget pressure) as separate fields,
  never collapsed into one score -- so the policy engine can react to *which*
  dimension is high, not just that something is.
- **Detectors** (NLI contradiction, P(True), a trained probe, injection/PII/
  toxicity classifiers) each produce one signal. They score; they don't
  decide.
- **The policy engine** decides. It reads RiskState against a versioned rule
  table (`policy_rules.yaml`) and returns one of ALLOW/VERIFY/RETRY/
  REGENERATE/REDACT/RESTRICT/BLOCK, deterministically. It never delegates a
  security decision to an LLM.
- **LangSmith** observes. It traces what happened for debugging; the agent
  never reads from it or blocks on it.

## Results

Baseline (bare Qwen2.5-7B-Instruct, single-shot) vs. the controlled agent
(full pipeline above), n=100 HaluEval `dialogue_data.json` test questions,
real generation, single seed. Full methodology and every number below:
[PHASE10_RESULTS.md](PHASE10_RESULTS.md).

- **Paired difference (factuality contradiction vs. knowledge, lower is
  better): +0.187, 95% CI [+0.113, +0.266]**, n=82 knowledge-clean subset,
  single seed.
- Secondary, all 100 examples including 18 with bad knowledge fields: +0.149,
  95% CI [+0.087, +0.214].
- Recovery-loop outcomes (n=60 interventions on the knowledge-clean subset),
  each a real rate with a Wilson 95% CI:
  - Succeeded: 45.0% [33.1%, 57.5%]
  - Improved but still flagged (exhausted retry budget): 18.3% [10.6%, 29.9%]
  - No meaningful change: 20.0% [11.8%, 31.8%]
  - **Made it worse: 16.7% [9.3%, 28.0%]**
- Cost: **8.5x mean latency** (9.07s -> 77.40s), **2.7x mean LLM calls**
  (1 -> 2.68), paid on every request, including the ones where the
  intervention didn't help or made things worse.
- Injection classifier false positives on 100 benign tool outputs: 1/100
  (1.0%, Wilson 95% CI [0.2%, 5.4%]).

Read plainly: the system helps on average, backfires on roughly one
intervention in six, and is expensive on every request whether or not it
helps. That combination -- not the improvement alone -- is the result.

## What didn't work

- **The combiner (probe + 3 P(True) variants + NLI, logistic regression)
  scored 0.91 val AUROC. The probe alone scored 0.94.** Standardizing
  features first (ruling out a scaling artifact) didn't change that. Adding
  the combiner made the factuality signal worse, not better, and it's
  reported that way rather than re-tuned until it looked competitive.
- **NLI contradiction scoring, in aggregate, scored 0.67 val AUROC --
  below a 0.73 length-only baseline.** It is precise on what it's designed
  for (catching a specific cross-clause self-contradiction, like the Marie
  Curie fixture used throughout development) but weak as a general-purpose
  aggregate detector.
- **HaluEval's `qa_data.json` has a severe length confound.** Its paired
  right-answer (~2 words) vs. hallucinated-answer (~11 words) responses let a
  two-feature length-only classifier hit 0.977 AUROC -- matching what looked
  like a near-perfect hidden-state probe, which turned out to be learning
  response length and punctuation, not fabrication. This forced a switch to
  `dialogue_data.json`, where the same length-only baseline drops to
  ~0.73-0.77. It's a finding about the benchmark, not just a bug fix.
- **18% of the HaluEval knowledge fields audited for Phase 10 were factually
  wrong** (11% confirmed, 7% probable, out of 100), and were excluded from
  the primary metric above. A benchmark's "ground truth" field needing its
  own accuracy audit is itself a data point about building factuality
  evaluations on top of existing datasets.

## Limitations

- **Single seed.** The paired confidence intervals above already exclude
  zero without one, but no individual regeneration's specific wording is
  itself statistically characterized -- only the aggregate effect across 100
  different questions is.
- **n=100.** A single stratified draw from the test split. Real, not
  simulated, but modest; a larger n would tighten every interval above.
- **Cross-model provenance.** HaluEval's `knowledge` fields come from its
  original construction pipeline, generated independently of the model used
  here (Qwen2.5-7B-Instruct) -- likely the source of some of the 18% error
  rate.
- **The vs-knowledge metric is bounded by knowledge-field reliability.**
  18% of those fields are wrong; results are reported on the knowledge-clean
  subset specifically to address this, but the audit that identified them is
  not independently verified ground truth.
- **The knowledge audit was performed by an LLM**, at confidence tiers
  (confirmed/probable/uncertain), checking HaluEval's claims against its own
  training knowledge -- not an independent human or database check. A
  project about the unreliability of LLM factuality judgments should not
  quietly treat its own audit as exempt from that same limitation.
- **Presidio's LOCATION entity type over-triggers** on real factuality-signal
  text; PERSON and LOCATION are excluded from the PII redaction set for this
  reason (see `src/detectors/detectors.py`).
- **`learning_loop.py`'s rule-update proposals are deliberately not
  applied.** Fitting policy thresholds on the same set used to evaluate the
  system would invalidate the results above.

## Repo layout

```
src/
  agent/       LangGraph, RiskState, router, policy engine, recovery, budget, cache
  detectors/   NLI, P(True), the trained probe's scoring path, combiner, complementarity
  data/        dataset loading, splitting, tokenization, scores-table export
kaggle/        GPU scripts (model load, probe train, P(True), eval) -- run on Kaggle, not locally
eval/          baseline comparison, log analysis, knowledge audit, learning loop, LangSmith demo
tests/         policy engine tests
artifacts/     probe weights and scores table (tracked); run logs (gitignored)
data/          HaluEval JSON files (gitignored, fetched at runtime)
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

PII detection (`presidio-analyzer`) needs a spaCy model, pulled separately from
pip -- the last line above fetches it.

## LangSmith tracing (optional)

Observability only -- the agent never reads from or blocks on it. To enable,
set these before running (never commit a key):

```powershell
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_API_KEY = "<your key>"
$env:LANGCHAIN_PROJECT = "control-planeai"
python eval/tracing_demo.py
```
