import random
import re
import statistics

from knowledge_audit import is_clean

EXAMPLE_RE = re.compile(
    r"--- example (\d+)/\d+ ---\n"
    r"BASELINE: .*?\n"
    r"  factuality contradiction vs knowledge: ([\d.]+), latency [\d.]+s\n"
    r"CONTROLLED: .*?\n"
    r"  policy: \{'action': '(\w+)'.*?\n"
    r"  factuality contradiction vs knowledge: ([\d.]+), latency [\d.]+s\n"
    r"  llm_calls_used: (\d+), retries_used: (\d+)",
    re.DOTALL,
)


def parse_log(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    results = []
    for m in EXAMPLE_RE.finditer(text):
        idx, base_f, action, ctrl_f, llm_calls, retries = m.groups()
        results.append(
            {
                "index": int(idx),
                "baseline_factuality": float(base_f),
                "controlled_factuality": float(ctrl_f),
                "action": action,
                "llm_calls": int(llm_calls),
                "retries": int(retries),
            }
        )
    return results


def bootstrap_ci(values: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return statistics.mean(values), lo, hi


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def main(log_path: str) -> None:
    results = parse_log(log_path)
    print(f"parsed {len(results)} examples from log")

    clean = [r for r in results if is_clean(r["index"])]
    dirty = [r for r in results if not is_clean(r["index"])]
    print(f"knowledge-clean: {len(clean)}, excluded (confirmed+probable bad knowledge): {len(dirty)}")

    print()
    print("=== PRIMARY: paired difference on knowledge-clean subset ===")
    diffs = [r["baseline_factuality"] - r["controlled_factuality"] for r in clean]  # positive = controlled better
    mean_diff, lo, hi = bootstrap_ci(diffs)
    print(f"n={len(clean)}, mean paired difference (baseline - controlled): {mean_diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("(positive = controlled has lower contradiction = better; CI excluding 0 = significant)")

    base_mean, base_lo, base_hi = bootstrap_ci([r["baseline_factuality"] for r in clean])
    ctrl_mean, ctrl_lo, ctrl_hi = bootstrap_ci([r["controlled_factuality"] for r in clean])
    print(f"baseline   mean: {base_mean:.4f}  95% CI [{base_lo:.4f}, {base_hi:.4f}]")
    print(f"controlled mean: {ctrl_mean:.4f}  95% CI [{ctrl_lo:.4f}, {ctrl_hi:.4f}]")

    print()
    print("=== SECONDARY: paired difference on all examples (including bad-knowledge) ===")
    diffs_all = [r["baseline_factuality"] - r["controlled_factuality"] for r in results]
    mean_diff_all, lo_all, hi_all = bootstrap_ci(diffs_all)
    print(f"n={len(results)}, mean paired difference: {mean_diff_all:+.4f}  95% CI [{lo_all:+.4f}, {hi_all:+.4f}]")

    print()
    print("=== disaggregated recovery-loop outcomes (knowledge-clean subset, intervened only) ===")
    intervened = [r for r in clean if r["llm_calls"] > 1]
    print(f"no intervention needed: {len(clean) - len(intervened)}/{len(clean)}")
    print(f"intervened: {len(intervened)}/{len(clean)}")

    succeeded = [r for r in intervened if r["action"] == "ALLOW"]
    still_flagged = [r for r in intervened if r["action"] != "ALLOW"]
    improved = [r for r in still_flagged if (r["baseline_factuality"] - r["controlled_factuality"]) > 0.05]
    worse = [r for r in still_flagged if (r["controlled_factuality"] - r["baseline_factuality"]) > 0.05]
    unchanged = [r for r in still_flagged if r not in improved and r not in worse]

    n_i = len(intervened)
    for label, group in [("succeeded (reached ALLOW)", succeeded), ("improved but still flagged", improved),
                          ("no meaningful change", unchanged), ("made worse", worse)]:
        p, lo, hi = wilson_ci(len(group), n_i)
        print(f"  {label}: {len(group)}/{n_i} = {p:.1%}  Wilson 95% CI [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main("artifacts/eval-v6-stdout.txt")
