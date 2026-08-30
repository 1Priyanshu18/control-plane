from dataclasses import dataclass
from pathlib import Path

import yaml

from analyze_eval import parse_log
from knowledge_audit import is_clean


@dataclass
class Proposal:
    rule_name: str
    observation: str
    suggested_change: str


def propose_updates(log_path: str) -> list[Proposal]:
    """Offline analysis of collected outcome data. Never writes anything --
    only proposes, backed by real observed numbers. A human decides whether
    any proposal is worth acting on; apply_update() is the only path that
    can ever touch policy_rules.yaml, and it refuses without explicit
    approval."""
    results = parse_log(log_path)
    clean = [r for r in results if is_clean(r["index"])]
    intervened = [r for r in clean if r["llm_calls"] > 1]

    proposals = []
    if intervened:
        worse = [
            r for r in intervened
            if (r["controlled_factuality"] - r["baseline_factuality"]) > 0.05 and r["action"] != "ALLOW"
        ]
        rate = len(worse) / len(intervened)
        if rate > 0.10:
            proposals.append(
                Proposal(
                    rule_name="regenerate_on_uncertainty / verify_high_factuality",
                    observation=(
                        f"of {len(intervened)} recovery-loop interventions in {log_path}, "
                        f"{len(worse)} ({rate:.1%}) made the response measurably worse"
                    ),
                    suggested_change=(
                        "consider lowering max_regenerations in recovery.py (currently 3) or "
                        "tightening the near-threshold escalation band in router.py -- the recovery "
                        "loop is not reliably improving outcomes at the current settings"
                    ),
                )
            )

        exhausted_no_improve = [
            r for r in intervened
            if r["retries"] >= 3 and r["action"] != "ALLOW"
            and abs(r["controlled_factuality"] - r["baseline_factuality"]) <= 0.05
        ]
        if len(exhausted_no_improve) / len(intervened) > 0.10:
            proposals.append(
                Proposal(
                    rule_name="recovery.max_regenerations",
                    observation=(
                        f"{len(exhausted_no_improve)}/{len(intervened)} interventions exhausted the "
                        "3-round retry budget with no meaningful change"
                    ),
                    suggested_change=(
                        "these rounds bought nothing but cost -- consider detecting 'no improvement "
                        "after round 1' and stopping early rather than spending all 3 rounds"
                    ),
                )
            )

    return proposals


def apply_update(policy_path: str, updates: dict, approved_by: str, reason: str) -> None:
    """The only function that may ever write policy_rules.yaml. Refuses
    without an explicit human approver -- this is the enforcement of
    'no automatic promotion', not just a documented convention."""
    if not approved_by or not approved_by.strip():
        raise ValueError("apply_update requires a non-empty approved_by -- no automatic promotion")
    if not reason or not reason.strip():
        raise ValueError("apply_update requires a non-empty reason")

    with open(policy_path) as f:
        policy = yaml.safe_load(f)

    policy["version"] += 1
    for rule in policy["rules"]:
        if rule["name"] in updates:
            rule.update(updates[rule["name"]])

    policy.setdefault("change_log", []).append(
        {"version": policy["version"], "approved_by": approved_by, "reason": reason}
    )

    with open(policy_path, "w") as f:
        yaml.safe_dump(policy, f, sort_keys=False)
