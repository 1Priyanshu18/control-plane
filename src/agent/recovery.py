from agent.budget import spend_retry
from agent.tools import search_tool
from detectors.nli import nli_contradiction_score

RECOVERABLE_ACTIONS = ("verify", "retry", "regenerate")


def recover(app, config: dict, max_regenerations: int = 3) -> tuple[dict, list[str]]:
    """Full spike -> rollback -> retrieve -> verify -> regenerate loop.
    Agent-state backtracking only: rolls back to an earlier checkpoint via
    the graph's own checkpointer (verified in graph.py / Phase 1), never
    anything at the token/neural level. Stops when the policy decision is no
    longer a recoverable MODIFY action, or retries_remaining hits 0."""
    trace: list[str] = []

    state = app.get_state(config).values
    decision = state["policy_decision"]
    trace.append(f"initial policy decision: {decision}")

    rounds = 0
    while decision["modify_action"] in RECOVERABLE_ACTIONS and rounds < max_regenerations:
        new_budget, allowed = spend_retry(state["budget"])
        if not allowed:
            trace.append("retries_remaining is 0 -- cannot regenerate further, stopping recovery loop")
            break

        trace.append(f"RISK SPIKE: {decision['reason']} (rule={decision['rule_name']})")

        evidence = search_tool(state["user_input"], poisoned=False)
        trace.append(f"RETRIEVE: evidence = {evidence!r}")

        verify_score = nli_contradiction_score({"knowledge": evidence, "response": state["llm_response"]})
        trace.append(f"VERIFY: response-vs-evidence contradiction score = {verify_score:.3f}")

        history = list(app.get_state_history(config))
        pre_llm_checkpoint = next(h for h in history if h.next == ("llm",))
        new_attempt = state["attempt"] + 1
        trace.append(
            f"ROLLBACK: reverting to the checkpoint before llm_node (was attempt={state['attempt']}), "
            f"retries_remaining {state['budget']['retries_remaining']} -> {new_budget['retries_remaining']}"
        )
        rollback_config = app.update_state(pre_llm_checkpoint.config, {"attempt": new_attempt, "budget": new_budget})
        # update_state()'s returned config carries only checkpoint state, not the
        # original invoke's tracing metadata -- without this, the regeneration
        # round traces as a separate, untagged LangSmith run instead of showing
        # up alongside the initial pass under the same tags/run_name.
        for key in ("tags", "run_name", "metadata"):
            if key in config:
                rollback_config[key] = config[key]

        trace.append(f"REGENERATE: resuming with attempt={new_attempt}")
        state = app.invoke(None, config=rollback_config)
        decision = state["policy_decision"]
        trace.append(f"post-regeneration policy decision: {decision}")
        rounds += 1

    trace.append(f"recovery loop finished after {rounds} regeneration(s); final action={decision['action']}")
    return state, trace
