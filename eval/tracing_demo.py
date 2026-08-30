import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langsmith import Client

from agent.budget import default_budget
from agent.graph import build_graph
from agent.recovery import recover
from config import load_config


def main() -> None:
    cfg = load_config()
    app = build_graph(cfg)

    budget = default_budget()
    budget["llm_calls_remaining"] = 5

    config = {
        "configurable": {"thread_id": "langsmith-demo"},
        "run_name": "control-planeai-intervention-demo",
        "tags": ["intervention", "phase9-demo"],
    }

    initial = app.invoke(
        {
            "user_input": "Tell me about the Eiffel Tower",
            "attempt": 2,  # the self-contradictory fixture response -- triggers an intervention
            "poisoned_tool": False,
            "budget": budget,
        },
        config=config,
    )
    print("initial policy_decision:", initial["policy_decision"])

    final_state, trace = recover(app, config)
    print("recovery trace:")
    for line in trace:
        print(" ", line)
    print("final policy_decision:", final_state["policy_decision"])

    print()
    print("waiting for traces to flush to LangSmith...")
    time.sleep(5)

    client = Client()
    runs = list(
        client.list_runs(
            project_name="control-planeai",
            filter='has(tags, "phase9-demo")',
            limit=20,
        )
    )
    print(f"found {len(runs)} traced runs tagged phase9-demo")
    for run in sorted(runs, key=lambda r: r.start_time):
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else None
        print(f"- run: {run.name} (id={run.id}) status={run.status} latency={latency}s")
        if run.name == "risk":
            print("  risk node outputs (7 dimensions, kept separate):")
            for key, value in (run.outputs or {}).get("risk", {}).items():
                print(f"    {key}: {value}")
        if run.name == "policy":
            print("  policy node outputs:", run.outputs)
        if run.name == "llm":
            print("  llm node outputs:", run.outputs)

    root_runs = [r for r in runs if r.parent_run_id is None]
    if root_runs:
        example = root_runs[0]
        print()
        print("trace URL:", client.get_run_url(run=example))


if __name__ == "__main__":
    main()
