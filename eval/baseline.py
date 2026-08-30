import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def length_features(rows: list[dict]) -> np.ndarray:
    return np.array([[len(r["response"]), len(r["response"].split())] for r in rows])


def length_baseline_auroc(train_rows: list[dict], val_rows: list[dict]) -> float:
    """Standing control: any detector metric reported alongside this is
    only meaningful if it clearly beats what response length alone predicts."""
    X_train = length_features(train_rows)
    y_train = np.array([r["label"] for r in train_rows])
    X_val = length_features(val_rows)
    y_val = np.array([r["label"] for r in val_rows])

    clf = LogisticRegression().fit(X_train, y_train)
    val_scores = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_scores)
