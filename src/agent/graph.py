from dataclasses import asdict
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.budget import BudgetState, spend_llm_call, spend_tool_call
from agent.cache import cache_get, cache_key, cache_set
from agent.generator import load_generator
from agent.policy import evaluate, load_policy
from agent.risk_state import RiskState
from agent.router import run_checks
from agent.tools import retrieval_tool, search_tool


class GraphState(TypedDict):
    user_input: str
    attempt: int
    poisoned_tool: bool
    knowledge: str
    budget: BudgetState
    cache_hit: bool
    llm_response: str
    tool_name: str
    tool_output: str
    risk: RiskState
    risk_log: list[str]
    policy_decision: dict
    final_response: str


def build_graph(config: dict, policy_path: str | None = None):
    generate = load_generator(config)
    policy = load_policy(policy_path) if policy_path else load_policy()

    def llm_node(state: GraphState) -> dict:
        attempt = state.get("attempt", 0)
        key = cache_key(state["user_input"], attempt)
        cached = cache_get(key)
        if cached is not None:
            return {"llm_response": cached, "cache_hit": True}

        budget, allowed = spend_llm_call(state["budget"], tokens=50, latency=0.5)
        if not allowed:
            return {"llm_response": "[BLOCKED: llm_calls budget exhausted]", "cache_hit": False, "budget": budget}

        response = generate(state["user_input"], attempt=attempt)
        cache_set(key, response)
        return {"llm_response": response, "cache_hit": False, "budget": budget}

    def tool_node(state: GraphState) -> dict:
        budget, allowed = spend_tool_call(state["budget"])
        if not allowed:
            return {"tool_output": "[BLOCKED: tool_calls budget exhausted]", "budget": budget}
        if state.get("knowledge"):
            return {"tool_name": "retrieval_tool", "tool_output": retrieval_tool(state["knowledge"]), "budget": budget}
        output = search_tool(state["user_input"], poisoned=state.get("poisoned_tool", False))
        return {"tool_name": "search_tool", "tool_output": output, "budget": budget}

    def risk_node(state: GraphState) -> dict:
        risk, log, budget = run_checks(
            generate,
            state["user_input"],
            state["llm_response"],
            state["tool_name"],
            state["tool_output"],
            state["budget"],
        )
        return {"risk": risk, "risk_log": log, "budget": budget}

    def policy_node(state: GraphState) -> dict:
        decision = evaluate(state["risk"], policy)
        return {"policy_decision": asdict(decision)}

    def response_node(state: GraphState) -> dict:
        return {"final_response": f"{state['llm_response']} ({state['tool_output']})"}

    graph = StateGraph(GraphState)
    graph.add_node("llm", llm_node)
    graph.add_node("tool", tool_node)
    graph.add_node("risk", risk_node)
    graph.add_node("policy", policy_node)
    graph.add_node("response", response_node)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", "tool")
    graph.add_edge("tool", "risk")
    graph.add_edge("risk", "policy")
    graph.add_edge("policy", "response")
    graph.add_edge("response", END)

    return graph.compile(checkpointer=MemorySaver())
