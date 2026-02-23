import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TARGET_COL

# Limit sample size for SHAP speed (TreeExplainer is fast but summary_plot can be heavy)
SHAP_SAMPLE = 500


def main():
    model = joblib.load(MODELS_DIR / "model.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_test = test[feature_cols]
    if len(X_test) > SHAP_SAMPLE:
        X_sample = X_test.sample(n=SHAP_SAMPLE, random_state=42)
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Summary plot (beeswarm)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP summary (impact on extreme_rain_tomorrow)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean |SHAP| = feature importance)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Feature importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP figures saved to {FIGURES_DIR} (shap_summary.png, shap_importance.png)")
    return shap_values


if __name__ == "__main__":
    main()
