from typing import Callable

from agent.budget import BudgetState, spend_verification_call
from agent.risk_state import RiskState, empty_risk_state
from agent.tools import TOOL_RISK_TABLE
from detectors.detectors import factuality_check, injection_check, pii_check, self_consistency_check, toxicity_check

NEAR_THRESHOLD_BAND = (0.2, 0.8)


def run_checks(
    generate: Callable[..., str],
    user_input: str,
    response: str,
    tool_name: str,
    tool_output: str,
    budget: BudgetState,
) -> tuple[RiskState, list[str], BudgetState]:
    """Cheap checks (PII, toxicity, injection-on-tool-output, tool-risk
    lookup, NLI self-contradiction) run every request. The expensive
    self-consistency check only fires when the cheap factuality signal lands
    in the near-threshold band -- and only if verification budget remains."""
    log: list[str] = []
    risk = empty_risk_state()

    pii_result = pii_check(response)
    risk["privacy_risk"] = pii_result
    entity_types = [e["entity_type"] for e in pii_result["entities"]]
    log.append(f"pii_check (cheap): score={pii_result['score']:.3f}, entities={entity_types}")

    tox_score = toxicity_check(response)
    risk["safety_risk"] = tox_score
    log.append(f"toxicity_check (cheap): {tox_score:.3f}")

    injection_score = injection_check(tool_output)
    risk["injection_risk"] = injection_score
    log.append(f"injection_check on TOOL OUTPUT (cheap): {injection_score:.3f}")

    risk["tool_risk"] = TOOL_RISK_TABLE.get(tool_name, "high")  # unknown tool defaults to high, fail safe
    log.append(f"tool_risk ({tool_name}): {risk['tool_risk']}")

    fact_score = factuality_check(response)
    risk["factuality_risk"] = fact_score
    log.append(f"factuality_check/nli (cheap): {fact_score:.3f}")

    low, high = NEAR_THRESHOLD_BAND
    if low <= fact_score <= high:
        budget, allowed = spend_verification_call(budget)
        if allowed:
            log.append(
                f"factuality score {fact_score:.3f} is within the near-threshold band {NEAR_THRESHOLD_BAND} "
                "-- escalating to self_consistency_check"
            )
            consistency_score = self_consistency_check(generate, user_input)
            risk["uncertainty_risk"] = consistency_score
            log.append(f"self_consistency_check (expensive): {consistency_score:.3f}")
        else:
            log.append(
                f"factuality score {fact_score:.3f} is within the near-threshold band {NEAR_THRESHOLD_BAND} "
                "-- would escalate, but verification_calls_remaining is 0 -- skipping"
            )
    else:
        log.append(
            f"factuality score {fact_score:.3f} is outside the near-threshold band {NEAR_THRESHOLD_BAND} "
            "-- skipping self_consistency_check"
        )

    return risk, log, budget
