from typing import Optional, TypedDict


class PIIEntity(TypedDict):
    entity_type: str
    start: int
    end: int
    score: float


class PrivacyRisk(TypedDict):
    """Presidio returns typed spans with offsets, not just a score -- kept
    intact here (not collapsed to a scalar) since Phase 8's redact
    intervention needs to know exactly which span to redact."""

    score: float
    entities: list[PIIEntity]


class RiskState(TypedDict):
    """Never collapsed to a scalar -- each dimension stays separate all the
    way through classify/decide/intervene. None means not yet assessed."""

    factuality_risk: Optional[float]
    uncertainty_risk: Optional[float]
    task_drift_risk: Optional[float]
    safety_risk: Optional[float]
    privacy_risk: Optional[PrivacyRisk]
    injection_risk: Optional[float]
    tool_risk: Optional[str]  # "low" | "high"


def empty_risk_state() -> RiskState:
    return RiskState(
        factuality_risk=None,
        uncertainty_risk=None,
        task_drift_risk=None,
        safety_risk=None,
        privacy_risk=None,
        injection_risk=None,
        tool_risk=None,
    )
