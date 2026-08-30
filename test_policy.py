from policy import evaluate, load_policy

POLICY = load_policy()


def _base_risk() -> dict:
    return {
        "factuality_risk": 0.0,
        "uncertainty_risk": None,
        "task_drift_risk": None,
        "safety_risk": 0.0,
        "privacy_risk": {"score": 0.0, "entities": []},
        "injection_risk": 0.0,
        "tool_risk": "low",
    }


def test_block_injection():
    risk = _base_risk()
    risk["injection_risk"] = 0.95
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "block_injection"
    assert decision.action == "BLOCK"


def test_block_toxic():
    risk = _base_risk()
    risk["safety_risk"] = 0.85
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "block_toxic"
    assert decision.action == "BLOCK"


def test_restrict_high_risk_tool():
    risk = _base_risk()
    risk["tool_risk"] = "high"
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "restrict_high_risk_tool"
    assert decision.action == "MODIFY"
    assert decision.modify_action == "restrict_tool"


def test_redact_pii():
    risk = _base_risk()
    risk["privacy_risk"] = {
        "score": 0.8,
        "entities": [{"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 10, "score": 0.8}],
    }
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "redact_pii"
    assert decision.action == "MODIFY"
    assert decision.modify_action == "redact"


def test_regenerate_on_uncertainty():
    risk = _base_risk()
    risk["uncertainty_risk"] = 0.6
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "regenerate_on_uncertainty"
    assert decision.action == "MODIFY"
    assert decision.modify_action == "regenerate"


def test_verify_high_factuality():
    risk = _base_risk()
    risk["factuality_risk"] = 0.85
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "verify_high_factuality"
    assert decision.action == "MODIFY"
    assert decision.modify_action == "verify"


def test_retry_moderate_factuality():
    risk = _base_risk()
    risk["factuality_risk"] = 0.5
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "retry_moderate_factuality"
    assert decision.action == "MODIFY"
    assert decision.modify_action == "retry"


def test_allow_default():
    risk = _base_risk()
    decision = evaluate(risk, POLICY)
    assert decision.rule_name == "allow_default"
    assert decision.action == "ALLOW"
