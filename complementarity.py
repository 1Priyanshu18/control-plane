import csv

THRESHOLD = 0.5
SIGNAL_COLUMNS = ["p_input_contradict", "p_self_contradict", "p_fact_contradict", "nli_contradiction"]


def main() -> None:
    with open("scores_table.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    val = [r for r in rows if r["split"] == "val"]
    for r in val:
        r["label"] = int(r["label"])
        r["probe"] = float(r["probe"])
        for c in SIGNAL_COLUMNS:
            r[c] = float(r[c])

    print(f"val set size: {len(val)}")

    fabrications = [r for r in val if r["label"] == 1]
    correct = [r for r in val if r["label"] == 0]
    print(f"fabrications (label=1): {len(fabrications)}, correct (label=0): {len(correct)}")

    probe_fn = [r for r in fabrications if r["probe"] <= THRESHOLD]  # probe misses fabrication
    probe_tp = [r for r in fabrications if r["probe"] > THRESHOLD]  # probe catches fabrication
    probe_fp = [r for r in correct if r["probe"] > THRESHOLD]  # probe wrongly flags correct response
    probe_tn = [r for r in correct if r["probe"] <= THRESHOLD]  # probe correctly clears it

    print()
    print(f"probe false negatives (misses fabrication): n={len(probe_fn)}")
    print(f"probe true positives (catches fabrication):  n={len(probe_tp)}")
    print(f"probe false positives (wrongly flags correct): n={len(probe_fp)}")
    print(f"probe true negatives (correctly clears correct): n={len(probe_tn)}")

    def flag_rate(group: list[dict], col: str) -> tuple[float, int]:
        if not group:
            return float("nan"), 0
        flagged = sum(1 for r in group if r[col] > THRESHOLD)
        return flagged / len(group), len(group)

    print()
    print("=== On probe's FALSE NEGATIVES (fabrications it scores clean) ===")
    print("does NLI / P(True) flag what the probe misses, vs. what it already catches?")
    for col in SIGNAL_COLUMNS:
        fn_rate, fn_n = flag_rate(probe_fn, col)
        tp_rate, tp_n = flag_rate(probe_tp, col)
        print(f"  {col}: flag rate on probe-FN (n={fn_n}) = {fn_rate:.3f}  vs  on probe-TP (n={tp_n}) = {tp_rate:.3f}")

    print()
    print("=== On probe's FALSE POSITIVES (correct responses it wrongly flags) ===")
    print("do other detectors also wrongly flag them (correlated error), or correctly stay quiet?")
    for col in SIGNAL_COLUMNS:
        fp_rate, fp_n = flag_rate(probe_fp, col)
        tn_rate, tn_n = flag_rate(probe_tn, col)
        print(f"  {col}: flag rate on probe-FP (n={fp_n}) = {fp_rate:.3f}  vs  on probe-TN (n={tn_n}) = {tn_rate:.3f}")


if __name__ == "__main__":
    main()
