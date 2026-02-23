import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR, TARGET_COL


def main():
    model = joblib.load(MODELS_DIR / "model.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_test = test[feature_cols]
    y_test = test[TARGET_COL]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        metrics["roc_auc"] = None  # only one class in y_test

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC curve (test set)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics (test set):", json.dumps(metrics, indent=2))
    print(f"Figures saved to {FIGURES_DIR}")
    return metrics


if __name__ == "__main__":
    main()
