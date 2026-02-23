import pandas as pd
import joblib
from pathlib import Path

from src.config import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, TARGET_COL

# XGBoost (algorithm not taught in lectures)
import xgboost as xgb


def load_feature_columns() -> list:
    """Read feature column names from processed data."""
    path = PROCESSED_DIR / "feature_columns.txt"
    if not path.exists():
        raise FileNotFoundError(f"Run preprocess first: {path}")
    return [line.strip() for line in path.read_text().strip().splitlines()]


def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    feature_cols = load_feature_columns()

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    # Handle class imbalance: scale_pos_weight = neg_count / pos_count
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
    print(f"Saved model to {MODELS_DIR / 'model.pkl'}")
    return model


if __name__ == "__main__":
    main()
