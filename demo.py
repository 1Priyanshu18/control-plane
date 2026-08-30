"""Live walkthrough of the control layer: three cases, real code, no GPU.

Uses the fake generator (deterministic, same output every run) so this is
safe to run on camera. Run: python demo.py
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.budget import default_budget
from agent.graph import build_graph
from agent.policy import evaluate, load_policy
from agent.recovery import recover
from agent.router import run_checks
from config import load_config
from fixtures import MARIE_CURIE_FIXTURE

TRACING_ENABLED = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "control-planeai")
DEMO_START_TIME = datetime.now(timezone.utc)

RISK_LABELS = [
    ("factuality_risk", "factuality_risk"),
    ("uncertainty_risk", "uncertainty_risk"),
    ("task_drift_risk", "task_drift_risk"),
    ("safety_risk", "safety_risk"),
    ("privacy_risk", "privacy_risk"),
    ("injection_risk", "injection_risk"),
    ("tool_risk", "tool_risk"),
]

BUDGET_FIELDS = [
    "llm_calls_remaining",
    "tool_calls_remaining",
    "retries_remaining",
    "verification_calls_remaining",
    "tokens_remaining",
    "latency_remaining",
]


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def subheader(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_risk_state(risk: dict) -> None:
    subheader("RiskState (seven dimensions, kept separate -- never collapsed to one score)")
    for label, key in RISK_LABELS:
        value = risk.get(key)
        if value is None:
            print(f"  {label:<18} not assessed")
        elif key == "privacy_risk":
            entities = [e["entity_type"] for e in value["entities"]]
            print(f"  {label:<18} score={value['score']:.3f}  entities={entities}")
            if entities and value["score"] == 0.0:
                print(
                    "                     (place/date/nationality/person-name entity types like these: "
                    "Presidio is known low-precision here, over-flagging ordinary encyclopedic text -- "
                    "excluded from the sensitive-entity set on purpose, see src/detectors/detectors.py; "
                    "score stays 0, no rule fires)"
                )
        elif isinstance(value, float):
            print(f"  {label:<18} {value:.3f}")
        else:
            print(f"  {label:<18} {value}")


def print_router_log(log: list[str]) -> None:
    subheader("Router: checks run vs. skipped")
    for line in log:
        print(f"  {line}")


def print_policy_decision(decision: dict) -> None:
    subheader("Policy decision")
    print(f"  action:        {decision['action']}")
    print(f"  modify_action: {decision['modify_action']}")
    print(f"  rule:          {decision['rule_name']}")
    print(f"  reason:        {decision['reason']}")


def print_budget_consumed(before: dict, after: dict) -> None:
    subheader("Budget consumed")
    for field in BUDGET_FIELDS:
        spent = before[field] - after[field]
        print(f"  {field:<26} {before[field]} -> {after[field]}  (spent {spent})")


def maybe_print_trace_url(tags: list[str]) -> None:
    if not TRACING_ENABLED:
        return
    subheader("LangSmith trace")
    try:
        import warnings

        from langsmith import Client

        print("  waiting for traces to flush...")
        time.sleep(3)
        client = Client()
        tag_filter = " and ".join(f'has(tags, "{t}")' for t in tags)
        with warnings.catch_warnings():
            # list_runs/get_run_url are deprecated (removal not until 2027) in favor
            # of client.runs.query/get_url, which need a project_id + trace_id lookup
            # first -- real added complexity for no near-term benefit. Silencing the
            # warning here rather than migrating.
            warnings.simplefilter("ignore", DeprecationWarning)
            runs = list(client.list_runs(project_name=LANGCHAIN_PROJECT, filter=tag_filter, limit=20))
            # scope to this execution only -- otherwise reruns (retakes) accumulate
            # every prior run's root traces under the same fixed tags/thread_id.
            root_runs = sorted(
                (r for r in runs if r.parent_run_id is None and r.start_time >= DEMO_START_TIME),
                key=lambda r: r.start_time,
            )
            if not root_runs:
                print("  no root run found yet (tracing may still be flushing)")
                return
            # a case with a recovery round produces two separate root invocations
            # (LangGraph doesn't nest one invoke() call under another) -- print all
            # of them, not just the latest, so the initial pass isn't hidden.
            noun = "invocation" if len(root_runs) == 1 else "invocations"
            print(f"  {len(root_runs)} traced {noun}:")
            for run in root_runs:
                print(f"    {client.get_run_url(run=run)}")
    except Exception as exc:
        print(f"  trace lookup failed: {exc}")


def pause(last: bool = False) -> None:
    if last:
        return
    input("\nPress Enter to continue...\n")


# ---------------------------------------------------------------------------
# Case 1: clean request, straight through
# ---------------------------------------------------------------------------


def case_1(app) -> None:
    header("CASE 1: clean request -- straight through")

    budget = default_budget()
    thread_config = {
        "configurable": {"thread_id": "demo-case-1"},
        "tags": ["control-planeai-demo", "case1"],
        "run_name": "demo-case-1-clean",
    }

    print("user_input: 'When was the Eiffel Tower completed?'")
    final_state = app.invoke(
        {
            "user_input": "When was the Eiffel Tower completed?",
            "attempt": 0,
            "poisoned_tool": False,
            "knowledge": "",
            "budget": budget,
        },
        config=thread_config,
    )

    print(f"\nresponse: {final_state['llm_response']!r}")
    print_risk_state(final_state["risk"])
    print_router_log(final_state["risk_log"])
    print_policy_decision(final_state["policy_decision"])

    subheader("Intervention")
    print("  none -- policy action is ALLOW, response passes straight through.")

    print_budget_consumed(budget, final_state["budget"])
    maybe_print_trace_url(["case1"])


# ---------------------------------------------------------------------------
# Case 2: the Marie Curie fixture -- detector comparison, honest negative
# finding included (the combiner does not beat the probe alone; the probe
# was NOT shown to structurally miss what NLI catches on this fixture --
# that original hypothesis did not hold up in Phase 3, so it is not staged
# here as if it did).
# ---------------------------------------------------------------------------


def case_2(policy: dict) -> None:
    header("CASE 2: the Marie Curie fixture -- detector comparison")

    knowledge = MARIE_CURIE_FIXTURE["knowledge"]
    response = MARIE_CURIE_FIXTURE["response"]
    print(f"knowledge: {knowledge!r}")
    print(f"response:  {response!r}")

    from agent.generator import _fake_generate as fake_generate

    budget = default_budget()
    risk, log, budget_after = run_checks(
        fake_generate, MARIE_CURIE_FIXTURE["context"], response, "retrieval_tool", knowledge, budget
    )
    decision = evaluate(risk, policy)

    print_risk_state(risk)
    print_router_log(log)
    print_policy_decision(decision.__dict__)

    subheader("Intervention")
    print(
        f"  the live agent would apply: {decision.action} -> {decision.modify_action} "
        f"(rule={decision.rule_name})."
    )
    print("  not executed here -- this case compares detectors on one fixture; see Case 3 for a live recovery loop.")

    print_budget_consumed(budget, budget_after)

    subheader("Detector comparison on THIS fixture")
    print("  nli_contradiction (live, computed above):        "
          f"{risk['factuality_risk']:.4f}  -- max pairwise contradiction across the response's own clauses")
    print("  probe (layer 17, hidden states from the real 7B model):")
    print("    not shown -- needs Qwen2.5-7B hidden states (GPU), cannot run with the fake generator,")
    print("    and no saved log has this fixture's exact score to cite instead of guessing.")
    print("  P(True) variants (Qwen2.5-7B forced-choice logit read) -- PRECOMPUTED, from artifacts/ptrue-v2-stdout.txt:")
    print("    p_input_contradict: 0.0015")
    print("    p_self_contradict:  0.9997")
    print("    p_fact_contradict:  0.0180")

    subheader("Phase 3 aggregate reference (val AUROC, not this fixture -- in-session analysis)")
    print("  probe (layer 17):            0.9398")
    print("  combiner (5 signals):        0.9072-0.9104")
    print("  nli_contradiction:           0.6711")
    print("  length-only baseline:        0.7339")
    print("  p_self_contradict:           0.5706")
    print("  p_fact_contradict:           0.4753")
    print("  p_input_contradict:          0.3515  (below chance)")
    print()
    print("  NEGATIVE FINDING: the combiner (0.91) did not beat the probe alone (0.94).")
    print("  Reported as-is, not re-tuned until it looked better.")

    if TRACING_ENABLED:
        subheader("LangSmith trace")
        print("  none -- this case calls run_checks/evaluate directly, bypassing the LangGraph")
        print("  graph, so there is nothing for LangSmith to trace here. See Case 1 or 3.")


# ---------------------------------------------------------------------------
# Case 3: recovery fires
# ---------------------------------------------------------------------------


def case_3(app) -> None:
    header("CASE 3: recovery loop -- retries, and here, succeeds")

    budget = default_budget()
    budget["llm_calls_remaining"] = 5  # need headroom for the initial call plus regenerations
    thread_config = {
        "configurable": {"thread_id": "demo-case-3"},
        "tags": ["control-planeai-demo", "case3"],
        "run_name": "demo-case-3-recovery",
    }

    print("user_input: 'Tell me about the Eiffel Tower'  (attempt=2 -- the fake generator's")
    print("self-contradictory canned response, chosen deliberately to trigger an intervention)")
    initial_state = app.invoke(
        {
            "user_input": "Tell me about the Eiffel Tower",
            "attempt": 2,
            "poisoned_tool": False,
            "knowledge": "",
            "budget": budget,
        },
        config=thread_config,
    )

    print(f"\ninitial response: {initial_state['llm_response']!r}")
    print_risk_state(initial_state["risk"])
    print_router_log(initial_state["risk_log"])
    print_policy_decision(initial_state["policy_decision"])

    final_state, trace = recover(app, thread_config)

    subheader("Recovery trace")
    for line in trace:
        print(f"  {line}")

    print(f"\nfinal response: {final_state['llm_response']!r}")
    print_risk_state(final_state["risk"])
    print_policy_decision(final_state["policy_decision"])

    subheader("Intervention")
    rounds = sum(1 for line in trace if line.startswith("RISK SPIKE"))
    print(f"  recovery loop executed {rounds} regeneration round(s); final action = {final_state['policy_decision']['action']}.")
    print(
        "  note: this fake-generator fixture set can only ever need 1 round to resolve -- only one of its"
    )
    print(
        "  three canned responses is self-contradictory, so 3-round budget exhaustion (a real, documented"
    )
    print(
        "  outcome in the Phase 10 GPU evaluation, 16.7% of interventions) cannot be reproduced live here."
    )

    print_budget_consumed(budget, final_state["budget"])
    maybe_print_trace_url(["case3"])


def main() -> None:
    cfg = load_config()
    assert cfg["generator"]["name"] == "fake", "demo.py requires the fake generator -- check config.yaml"
    policy = load_policy("policy_rules.yaml")
    app = build_graph(cfg, policy_path="policy_rules.yaml")

    print("control_planeAI -- live demo")
    print("generator: fake (deterministic, no GPU)")
    print(f"LangSmith tracing: {'enabled' if TRACING_ENABLED else 'disabled'}")

    case_1(app)
    pause()
    case_2(policy)
    pause()
    case_3(app)
    pause(last=True)

    print()
    print("done.")


if __name__ == "__main__":
    main()
