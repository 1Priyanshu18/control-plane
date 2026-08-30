# control_planeAI

Large language models state false things confidently, in the same tone as
true things. The usual fix is a single detector that checks the output and
either passes it or blocks it. That's fragile: one detector's mistake
becomes the whole system's mistake, and there's no way to react differently
to "this looks made up" versus "this contains a phone number" versus "this
came from a tool that shouldn't be trusted."

control_planeAI is a control layer that sits around an LLM agent instead:
it watches several different kinds of risk at once, runs the checks that
are actually relevant to the request, and only then decides what to do --
let the response through, fix it, or block it.

## Architecture

Five pieces, kept deliberately separate:

LangGraph runs the agent as a sequence of steps and saves the state after
each one. That's what makes recovery possible later: if something goes
wrong, the system can rewind to the point right before the bad response was
generated and try again, instead of patching broken text after the fact.

RiskState is a record of seven separate risk signals for the current
request: factuality (is the response likely to contain a fabricated claim),
uncertainty (did the model contradict itself when asked twice), task_drift
(reserved for detecting off-task responses -- not yet implemented, always
empty in this version), safety (toxicity), privacy (personal information in
the response), injection (a prompt-injection attack hidden in a tool's
output), and tool (whether the tool just used is safe or requires
authorization). These stay separate all the way through the pipeline. They
are never averaged or combined into one score, because a policy that only
sees "risk: 0.6" can't tell a mildly uncertain answer from a serious privacy
leak.

Uncertainty is implemented, not skipped by omission: checking it means
asking the model the same question twice and seeing if it disagrees with
itself, which costs an extra call to the model. That's only worth paying
for when the cheap factuality check is inconclusive. If factuality already
lands confidently low (clearly fine) or confidently high (clearly not), the
router skips the uncertainty check entirely and it stays unassessed for
that request. This is the adaptive part of the design working as intended,
not a gap -- it just means uncertainty shows up as "not assessed" more
often than a reader might expect from a dimension that's fully built.

Detectors produce those signals. Some are simple: a keyword-trained
classifier for toxicity, a lookup table for which tools are risky. Others
are more involved and worth naming, because they come up again in the
results below:

- NLI, short for natural language inference, is a separate pretrained model
  whose job is to judge whether two pieces of text contradict each other.
  Here it's used to check the response against itself (do two parts of the
  answer disagree) and, where available, against a reference fact.
- P(True) asks the model a yes/no question about its own answer -- "is this
  true?" -- but instead of trusting the model's typed-out reply, it reads
  the internal probability the model assigned to "true" versus "false"
  before generating any text. The idea is that the number might be more
  honest than the model's stated answer.
- A probe is a small classifier trained to read the model's internal
  activations (the numeric state inside the model while it's generating a
  response) and predict, directly from those numbers, whether the response
  is likely to be wrong -- without needing the model to say anything about
  it.

None of the detectors decide anything. They only produce a number or a
label; the decision is made separately.

The policy engine decides. It reads RiskState against a table of rules in
`policy_rules.yaml` -- first matching rule wins, no exceptions, no learning,
no LLM call involved in the decision itself. Every decision is one of three
actions: ALLOW (send the response through), BLOCK (refuse it outright,
reserved for the clearest cases like a strong injection match), or MODIFY.
MODIFY carries a second field naming what kind of fix applies: verify
(the factuality signal is high enough to check the claim against outside
evidence before it goes out), retry (the factuality signal is moderately
elevated -- below verify's threshold -- and triggers the same
retrieve-and-regenerate loop), regenerate (the model disagreed with itself
across two samples), redact (strip out flagged personal information), or
restrict_tool (block a side-effecting tool call pending authorization). The
rule table is versioned, so a change to a threshold is a reviewable diff,
not a silent edit.

LangSmith observes. It records a trace of what happened, for debugging
after the fact. The agent never reads from it and never waits on it -- it
has no way to affect a live decision.

## A worked example

The clearest way to see the loop is to watch one request go through it.

1. The model's first response contradicts itself in one sentence: it says
   the Eiffel Tower's construction was fully government-funded, then in the
   same breath says Gustave Eiffel paid for it privately.
2. Policy decision: MODIFY, verify -- the self-contradiction score crossed
   the threshold that means "don't send this yet, check it first."
3. RISK SPIKE -- the recovery loop picks up the flagged decision and starts.
4. RETRIEVE -- it pulls a reference answer from a search tool: a plain,
   uncontroversial description of the Eiffel Tower.
5. VERIFY -- it compares the response against that reference. Contradiction
   score: 1.000, the maximum.
6. ROLLBACK -- using LangGraph's saved checkpoint, execution rewinds to the
   point right before the bad response was generated.
7. REGENERATE -- the model is asked again.
8. The new response: a plain, uncontested sentence about when the tower was
   completed. No self-contradiction.
9. Policy decision: ALLOW.
10. Recovery finished after one round. This is the response that reaches
    the user, not the first one.

## Results

These numbers compare a plain agent (the base model, single response, no
checks) against the controlled agent above, on 100 questions from HaluEval
-- a public benchmark of model responses labeled as containing a fabricated
claim or not. For the dialogue data used here, that labeling wasn't done by
human reviewers: HaluEval's own construction process used ChatGPT to
generate candidate fabricated responses, then a second ChatGPT call to pick
the most plausible-sounding wrong one. Given that, an 18% error rate in
these reference answers (found below) isn't a surprising anomaly -- it's
close to what you'd expect from labels one language model assigned to
another's output, unchecked by a person. Both agents answer the same 100
questions; nothing here is simulated. Full methodology and every number:
[PHASE10_RESULTS.md](PHASE10_RESULTS.md).

The main metric is how much a response contradicts a reference answer, on
a 0-to-1 scale from a contradiction-detection model (0 means no
contradiction found, 1 means certain contradiction) -- lower is better.
Comparing the two agents on the same question and averaging the difference
is called a paired difference; it's a more sensitive comparison than
averaging each agent's scores separately, because each question serves as
its own baseline.

18 of the 100 reference answers turned out to be factually wrong themselves
(more on that below), so the primary result uses the 82 clean ones. On
those, the plain agent's responses contradicted the reference with an
average score of 0.616; the controlled agent's averaged 0.429. That's a
paired difference of **+0.187**, with a 95% confidence interval of [+0.113,
+0.266] -- an interval that does not include zero, so this is a real
effect, not noise. It's roughly a 30% relative reduction in contradiction
score. Across all 100 questions, including the 18 with bad references, the
difference is smaller but still real: +0.149, CI [+0.087, +0.214].

That average hides a wide spread in what recovery actually does on any
given request. Of the 60 cases (within the clean 82) where the recovery
loop fired at all, the outcomes, each with its own confidence interval:

- Succeeded, reached ALLOW: 45.0% [33.1%, 57.5%]
- Improved but still flagged when it ran out of retries: 18.3% [10.6%, 29.9%]
- No meaningful change: 20.0% [11.8%, 31.8%]
- Made the response worse: **16.7%** [9.3%, 28.0%]

None of this is free. Average latency goes from 9.07 seconds for the plain
agent to 77.40 seconds for the controlled one -- **8.5 times** as long --
and the controlled agent makes 2.7 times as many calls to the model. That
cost is paid on every request, including the one-in-six where the
intervention actually made the response worse.

Separately, the injection detector was checked against all 100 real tool
outputs and flagged 1 of them as an attack when it wasn't one (1.0% false
positive rate, CI [0.2%, 5.4%]).

Read together: the system helps on average, backfires on roughly one
intervention in six, and costs several times more than doing nothing on
every single request, whether or not the intervention helps. That
combination, not the average improvement by itself, is the actual result.

## What didn't work

A combiner -- one classifier trained on top of the probe, the three P(True)
variants, and NLI, to fuse all five factuality-related scores into a single
number -- was built to test a real architectural question. The policy
engine reads one number per RiskState dimension, not five; something has
to turn five factuality signals into one factuality_risk value. Today
that value comes from NLI alone (the router's cheap self-contradiction
check), and the combiner exists to ask whether it should come from
something more sophisticated instead. That question is worth asking
regardless of the answer -- fusing multiple signals into one decision-ready
number is a structural need, not just an accuracy experiment.

The answer, measured, was no. The combiner scored 0.91 on AUROC, a
standard way to score how well a detector separates good responses from
bad ones, running from 0.5 (no better than a coin flip) to 1.0 (perfect
separation). The probe alone scored 0.94. Combining signals made this
worse, not better, even after standardizing the inputs to rule out a
scaling artifact. A separate analysis dug into why: examples were split
into groups by whether the probe got them right or wrong, then checked
whether NLI (or the other signals) flagged what the probe missed. It
didn't -- NLI flagged 76% of the fabrications the probe caught, but only
21% of the ones it missed. If NLI were catching what the probe misses,
that number would run the other way. The signals aren't complementary
here; they're mostly redundant, and a combiner built from redundant
signals inherits their shared blind spots instead of covering for them.
This is why the combiner stayed a standalone evaluation script
(`src/detectors/combiner.py`) and never replaced the live NLI check --
it didn't earn the added complexity.

That's a real limit, but it's a narrow one: within factuality, on this
dataset, five signals didn't beat one. It says nothing about whether
RiskState's seven dimensions should be separate -- injection risk and
privacy risk catch entirely different failures and neither substitutes
for factuality risk. The redundancy found here is specific to these five
factuality detectors on this dataset, not a verdict on keeping signals
separate in general.

NLI's contradiction score, checked in aggregate across many examples,
scored 0.67 AUROC -- below a length-only baseline of 0.73. A length-only
baseline is a classifier that looks at nothing but how long the response
is; it exists as a sanity check for whether a detector is picking up on
something real, or just learning that longer answers tend to be wrong more
often. NLI beats that baseline on the one thing it was built for -- catching
a specific kind of self-contradiction, like the Eiffel Tower example above
-- but is a weak general-purpose detector on average.

A more basic problem showed up first, in an earlier evaluation. One part
of HaluEval (its `qa_data.json` file) pairs a short correct answer against
a much longer fabricated one, so a classifier that looks only at response
length scores 0.977 AUROC on it -- almost perfect, and almost exactly what
a real hidden-state probe scored on the same data. The probe wasn't
detecting fabrication; it was detecting length. Switching to a different
part of the benchmark (`dialogue_data.json`) dropped the length-only score
to around 0.73-0.77, closer to what a length baseline should look like.
This is a finding about the benchmark itself, not just a bug that got
fixed.

And **18%** of the HaluEval reference answers used for the Phase 10 evaluation
above turned out to be factually wrong on inspection (11% clearly, 7%
probably), and were excluded from the primary result. A benchmark's
reference answers needing their own accuracy check is itself worth noting,
in a project about whether language models can be trusted to state facts
correctly.

## Limitations

Everything above comes from a single run with one random seed. The paired
confidence intervals already rule out pure noise, but the exact wording any
individual regeneration produces isn't itself statistically characterized
-- only the aggregate effect over 100 different questions is. The sample
itself is a single draw of 100 questions from the test set; real, not
simulated, but modest, and a larger sample would tighten every interval
above.

HaluEval's reference answers were written independently of the model used
here (Qwen2.5-7B-Instruct) as part of the benchmark's own construction --
likely part of why 18% of them turned out to be wrong. Because of that
error rate, the main metric is only as trustworthy as the reference text
it's compared against; results are reported on the error-free subset
specifically to address this, but the audit that identified the errors is
itself not independently verified ground truth: it was performed by an
LLM, at three confidence tiers, checking HaluEval's claims against its own
training knowledge, not by a human or a database lookup. A project about
whether language models can be trusted to judge factuality shouldn't quietly
exempt its own audit from that same question.

Presidio, the personal-information detector used here, over-flags on
ordinary encyclopedic text -- it will tag "Paris" or "Gustave Eiffel" the
same way it tags a phone number. Person and location entity types are
excluded from the set that actually triggers a redaction for this reason
(see `src/detectors/detectors.py`).

`eval/learning_loop.py` can propose changes to the policy's rule
thresholds based on what recovery actually did, but those proposals are
deliberately never applied automatically here. Tuning the rules on the
same data used to evaluate the system would invalidate the results above.

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
demo.py        a runnable, deterministic walkthrough of the worked example above
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

Once installed, `python demo.py` runs the worked example above (and two
others) end to end, using a fixed fake model response so the output is the
same every time -- no GPU needed.

## LangSmith tracing (optional)

Observability only -- the agent never reads from or blocks on it. To enable,
set these before running (never commit a key):

```powershell
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_API_KEY = "<your key>"
$env:LANGCHAIN_PROJECT = "control-planeai"
python eval/tracing_demo.py
```
