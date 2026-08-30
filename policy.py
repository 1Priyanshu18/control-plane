from dataclasses import dataclass
from pathlib import Path

import yaml

_OPERATORS = {
    "gte": lambda value, threshold: value is not None and value >= threshold,
    "lte": lambda value, threshold: value is not None and value <= threshold,
    "gt": lambda value, threshold: value is not None and value > threshold,
    "lt": lambda value, threshold: value is not None and value < threshold,
    "eq": lambda value, threshold: value == threshold,
    "always": lambda value, threshold: True,
}

DEFAULT_POLICY_PATH = "policy_rules.yaml"


@dataclass
class PolicyDecision:
    action: str  # "ALLOW" | "MODIFY" | "BLOCK"
    modify_action: str | None
    reason: str
    rule_name: str


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_risk(risk: dict) -> dict:
    """Policy rules match on simple named fields. This does not collapse
    RiskState's dimensions into one score -- it just extracts each
    dimension's own scalar (or a named sub-field, for privacy_risk's nested
    score) for independent rule matching."""
    privacy = risk.get("privacy_risk")
    return {
        "factuality_risk": risk.get("factuality_risk"),
        "uncertainty_risk": risk.get("uncertainty_risk"),
        "task_drift_risk": risk.get("task_drift_risk"),
        "safety_risk": risk.get("safety_risk"),
        "privacy_risk_score": privacy["score"] if privacy else None,
        "injection_risk": risk.get("injection_risk"),
        "tool_risk": risk.get("tool_risk"),
    }


def evaluate(risk: dict, policy: dict) -> PolicyDecision:
    """Deterministic, first-match-wins over the versioned rule table. No LLM
    call, no randomness -- purely a function of risk and the config."""
    flat = flatten_risk(risk)
    for rule in policy["rules"]:
        field_value = flat.get(rule["field"])
        op = _OPERATORS[rule["operator"]]
        if op(field_value, rule["threshold"]):
            return PolicyDecision(
                action=rule["action"],
                modify_action=rule.get("modify_action"),
                reason=rule["reason"],
                rule_name=rule["name"],
            )
    raise RuntimeError("no rule matched -- policy table must end with a catch-all rule")
