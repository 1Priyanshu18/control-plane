import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SIGNAL_COLUMNS = ["probe", "p_input_contradict", "p_self_contradict", "p_fact_contradict", "nli_contradiction"]


def load_scores_table(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        for col in SIGNAL_COLUMNS:
            r[col] = float(r[col])
    return rows


def load_response_lengths(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["response_id"]: {"chars": len(r["response"]), "words": len(r["response"].split())} for r in rows}


def _xy(rows: list[dict], columns: list[str], lengths: dict[str, dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    for r in rows:
        row_feats = [r[c] for c in columns]
        if lengths is not None:
            row_feats += [lengths[r["response_id"]]["chars"], lengths[r["response_id"]]["words"]]
        feats.append(row_feats)
    X = np.array(feats)
    y = np.array([r["label"] for r in rows])
    return X, y


def main() -> None:
    rows = load_scores_table("scores_table.csv")
    lengths = load_response_lengths("artifacts/scores_partial.csv")

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    print(f"train: {len(train_rows)}, val: {len(val_rows)}")

    print()
    print("individual signals (raw score AUROC on val, no fitting needed):")
    y_val = np.array([r["label"] for r in val_rows])
    for col in SIGNAL_COLUMNS:
        scores = np.array([r[col] for r in val_rows])
        auroc = roc_auc_score(y_val, scores)
        print(f"  {col}: {auroc:.4f}")

    length_X_train = np.array([[lengths[r["response_id"]]["chars"], lengths[r["response_id"]]["words"]] for r in train_rows])
    length_X_val = np.array([[lengths[r["response_id"]]["chars"], lengths[r["response_id"]]["words"]] for r in val_rows])
    y_train = np.array([r["label"] for r in train_rows])
    length_clf = LogisticRegression().fit(length_X_train, y_train)
    length_auroc = roc_auc_score(y_val, length_clf.predict_proba(length_X_val)[:, 1])
    print(f"  length_only (fitted): {length_auroc:.4f}")

    print()
    print("combiner (logistic regression over the 5 detector signals):")
    X_train, y_train = _xy(train_rows, SIGNAL_COLUMNS)
    X_val, y_val = _xy(val_rows, SIGNAL_COLUMNS)
    combiner = LogisticRegression().fit(X_train, y_train)
    combiner_auroc = roc_auc_score(y_val, combiner.predict_proba(X_val)[:, 1])
    print(f"  combiner (5 signals): {combiner_auroc:.4f}")
    print(f"  coefficients: {dict(zip(SIGNAL_COLUMNS, combiner.coef_[0].round(3)))}")

    scaler = StandardScaler().fit(X_train)
    combiner_scaled = LogisticRegression().fit(scaler.transform(X_train), y_train)
    combiner_scaled_auroc = roc_auc_score(y_val, combiner_scaled.predict_proba(scaler.transform(X_val))[:, 1])
    print(f"  combiner (5 signals, standardized features): {combiner_scaled_auroc:.4f}")

    print()
    print("ablation: combiner with length as a fourth arm (5 signals + char_len + word_count):")
    X_train2, y_train2 = _xy(train_rows, SIGNAL_COLUMNS, lengths)
    X_val2, y_val2 = _xy(val_rows, SIGNAL_COLUMNS, lengths)
    combiner2 = LogisticRegression().fit(X_train2, y_train2)
    combiner2_auroc = roc_auc_score(y_val2, combiner2.predict_proba(X_val2)[:, 1])
    print(f"  combiner (5 signals + length): {combiner2_auroc:.4f}")
    print(f"  delta vs signals-only combiner: {combiner2_auroc - combiner_auroc:+.4f}")


if __name__ == "__main__":
    main()
