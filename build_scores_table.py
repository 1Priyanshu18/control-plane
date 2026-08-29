import csv

from nli import nli_contradiction_score

IN_PATH = "artifacts/scores_partial.csv"
OUT_PATH = "scores_table.csv"


def main() -> None:
    with open(IN_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "response_id",
                "split",
                "label",
                "probe",
                "p_input_contradict",
                "p_self_contradict",
                "p_fact_contradict",
                "nli_contradiction",
            ]
        )
        for i, r in enumerate(rows):
            nli_score = nli_contradiction_score({"knowledge": r["knowledge"], "response": r["response"]})
            writer.writerow(
                [
                    r["response_id"],
                    r["split"],
                    r["label"],
                    r["probe"],
                    r["p_input_contradict"],
                    r["p_self_contradict"],
                    r["p_fact_contradict"],
                    nli_score,
                ]
            )
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)}")

    print(f"wrote {OUT_PATH} with {len(rows)} rows")


if __name__ == "__main__":
    main()
