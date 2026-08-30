import torch

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device available")

_device_count = torch.cuda.device_count()
_device_name = torch.cuda.get_device_name(0)
_major, _minor = torch.cuda.get_device_capability(0)
_capability = _major + _minor / 10

print(f"device count: {_device_count}")
print(f"device name: {_device_name}")
print(f"compute capability: {_major}.{_minor}")

if _capability < 7.5:
    raise SystemExit(
        f"GPU compute capability {_major}.{_minor} ({_device_name}) is below the 7.5 "
        "minimum bitsandbytes 4-bit quantization requires; aborting before weight download."
    )

import os
import random
import statistics
import subprocess
import sys
import time

subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes", "presidio-analyzer", "spacy"], check=True)
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg", "-q"], check=True)

CODE_DIR = None
for dirpath, _dirnames, filenames in os.walk("/kaggle/input"):
    if "config.py" in filenames:
        CODE_DIR = dirpath
        break
if CODE_DIR is None:
    raise RuntimeError(f"could not find config.py under /kaggle/input; tree: {list(os.walk('/kaggle/input'))}")
sys.path.insert(0, CODE_DIR)

from budget import default_budget
from config import load_config
from data import flatten_dialogue, load_dialogue
from generator import _real_generate
from graph import build_graph
from nli import nli_contradiction_score
from recovery import recover
from split import response_level_split


def sample_test_groups(n: int) -> list[dict]:
    records = load_dialogue()
    flat = flatten_dialogue(records)
    splits = response_level_split(flat)
    test = splits["test"]
    seen: dict[str, dict] = {}
    for r in test:
        if r["group_id"] not in seen:
            seen[r["group_id"]] = r
    groups = list(seen.values())
    return random.sample(groups, min(n, len(groups)))


def main(config: dict, n_examples: int) -> None:
    print(f"TEST SET FIRST TOUCH: sampling {n_examples} groups from the test split")
    groups = sample_test_groups(n_examples)

    real_config = dict(config)
    real_config["generator"] = {"name": "real"}
    app = build_graph(real_config, policy_path=os.path.join(CODE_DIR, "policy_rules.yaml"))

    results = []
    for i, g in enumerate(groups):
        print(f"\n--- example {i + 1}/{len(groups)} ---")
        prompt = g["context"]
        knowledge = g["knowledge"]

        t0 = time.time()
        baseline_response = _real_generate(prompt, attempt=0)
        baseline_latency = time.time() - t0
        baseline_factuality = nli_contradiction_score({"knowledge": knowledge, "response": baseline_response})
        print(f"BASELINE: {baseline_response!r}")
        print(f"  factuality contradiction vs knowledge: {baseline_factuality:.3f}, latency {baseline_latency:.1f}s")

        thread_config = {"configurable": {"thread_id": f"eval-{i}"}}
        budget = default_budget()
        budget["llm_calls_remaining"] = 5

        t0 = time.time()
        app.invoke(
            {
                "user_input": prompt,
                "attempt": 0,
                "poisoned_tool": False,
                "knowledge": knowledge,
                "budget": budget,
            },
            config=thread_config,
        )
        final_state, trace = recover(app, thread_config)
        controlled_latency = time.time() - t0
        controlled_factuality = nli_contradiction_score(
            {"knowledge": knowledge, "response": final_state["llm_response"]}
        )
        print(f"CONTROLLED: {final_state['llm_response']!r}")
        print(f"  policy: {final_state['policy_decision']}")
        print(
            f"  factuality contradiction vs knowledge: {controlled_factuality:.3f}, "
            f"latency {controlled_latency:.1f}s"
        )
        llm_calls_used = 5 - final_state["budget"]["llm_calls_remaining"]
        retries_used = 3 - final_state["budget"]["retries_remaining"]
        print(f"  llm_calls_used: {llm_calls_used}, retries_used: {retries_used}")

        results.append(
            {
                "baseline_factuality": baseline_factuality,
                "baseline_latency": baseline_latency,
                "controlled_factuality": controlled_factuality,
                "controlled_latency": controlled_latency,
                "controlled_llm_calls": llm_calls_used,
                "controlled_retries": retries_used,
                "controlled_action": final_state["policy_decision"]["action"],
            }
        )

    print()
    print("=" * 70)
    print(f"AGGREGATE (n={len(results)}, 1 seed, smoke scale -- not a full evaluation)")
    print(
        f"baseline   mean factuality-contradiction: "
        f"{statistics.mean(r['baseline_factuality'] for r in results):.3f}"
    )
    print(
        f"controlled mean factuality-contradiction: "
        f"{statistics.mean(r['controlled_factuality'] for r in results):.3f}"
    )
    print(f"baseline   mean latency: {statistics.mean(r['baseline_latency'] for r in results):.2f}s")
    print(f"controlled mean latency: {statistics.mean(r['controlled_latency'] for r in results):.2f}s")
    print(f"controlled mean llm_calls used: {statistics.mean(r['controlled_llm_calls'] for r in results):.2f}")
    print(f"controlled mean retries used: {statistics.mean(r['controlled_retries'] for r in results):.2f}")
    action_counts: dict[str, int] = {}
    for r in results:
        action_counts[r["controlled_action"]] = action_counts.get(r["controlled_action"], 0) + 1
    print(f"controlled policy action distribution: {action_counts}")


if __name__ == "__main__":
    cfg = load_config(os.path.join(CODE_DIR, "config.yaml"))
    random.seed(cfg["seed"])
    main(cfg, n_examples=100)
