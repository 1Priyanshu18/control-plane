from typing import TypedDict


class BudgetState(TypedDict):
    llm_calls_remaining: int
    tool_calls_remaining: int
    retries_remaining: int
    verification_calls_remaining: int
    tokens_remaining: int
    latency_remaining: float


def default_budget() -> BudgetState:
    return BudgetState(
        llm_calls_remaining=2,
        tool_calls_remaining=5,
        retries_remaining=3,
        verification_calls_remaining=5,
        tokens_remaining=2000,
        latency_remaining=30.0,
    )


def spend_llm_call(budget: BudgetState, tokens: int, latency: float) -> tuple[BudgetState, bool]:
    """Returns (new_budget, allowed). Never mutates the input."""
    if budget["llm_calls_remaining"] <= 0:
        return budget, False
    new_budget = dict(budget)
    new_budget["llm_calls_remaining"] -= 1
    new_budget["tokens_remaining"] = max(0, budget["tokens_remaining"] - tokens)
    new_budget["latency_remaining"] = max(0.0, budget["latency_remaining"] - latency)
    return new_budget, True


def spend_tool_call(budget: BudgetState) -> tuple[BudgetState, bool]:
    if budget["tool_calls_remaining"] <= 0:
        return budget, False
    new_budget = dict(budget)
    new_budget["tool_calls_remaining"] -= 1
    return new_budget, True


def spend_verification_call(budget: BudgetState) -> tuple[BudgetState, bool]:
    if budget["verification_calls_remaining"] <= 0:
        return budget, False
    new_budget = dict(budget)
    new_budget["verification_calls_remaining"] -= 1
    return new_budget, True


def spend_retry(budget: BudgetState) -> tuple[BudgetState, bool]:
    if budget["retries_remaining"] <= 0:
        return budget, False
    new_budget = dict(budget)
    new_budget["retries_remaining"] -= 1
    return new_budget, True
