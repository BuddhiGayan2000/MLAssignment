import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOCATION_NAMES = {0: "Kandy (Central)", 1: "Batticaloa (Eastern)", 2: "Anuradhapura (North Central)"}


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    if not model_path.exists():
        return None, None
    model = joblib.load(model_path)
    fc_path = PROJECT_ROOT / "models" / "feature_columns.pkl"
    feature_cols = joblib.load(fc_path) if fc_path.exists() else []
    return model, feature_cols


def main():
    st.set_page_config(page_title="Extreme Rain Forecast", page_icon="🌧️", layout="centered")
    st.title("🌧️ Extreme Rain Tomorrow — Sri Lanka Agriculture")
    st.caption("ML Assignment: Predict high flood-risk days (e.g. >50 mm next day) from weather. Data: Open-Meteo; model: XGBoost.")

    model, feature_cols = load_model()
    if model is None:
        st.error("Model not found. Train first: `python -m src.train` from project root.")
        return

    # Sidebar: input mode
    mode = st.sidebar.radio("Input mode", ["Manual input", "Sample from test set"], index=0)
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        test_path = PROCESSED_DIR / "test.csv"

    if mode == "Sample from test set" and test_path.exists():
        test_df = pd.read_csv(test_path)
        row_idx = st.sidebar.slider("Row index (test set)", 0, min(500, len(test_df) - 1), 0)
        sample = test_df.iloc[row_idx]
        st.sidebar.write("Selected row date:", sample.get("date", "—"))
        inputs = {c: float(sample[c]) for c in feature_cols if c in sample.index}
    else:
        # Manual inputs
        st.subheader("Weather features (today)")
        loc_id = st.selectbox("Location", options=[0, 1, 2], format_func=lambda x: LOCATION_NAMES[x], index=0)
        col1, col2 = st.columns(2)
        with col1:
            temp_max = st.number_input("Max temperature (°C)", value=30.0, min_value=15.0, max_value=45.0, step=0.5)
            temp_min = st.number_input("Min temperature (°C)", value=24.0, min_value=15.0, max_value=40.0, step=0.5)
            precip = st.number_input("Precipitation today (mm)", value=5.0, min_value=0.0, max_value=200.0, step=0.5)
            roll_3d = st.number_input("Precipitation 3-day sum (mm)", value=10.0, min_value=0.0, max_value=400.0, step=1.0)
        with col2:
            roll_7d = st.number_input("Precipitation 7-day sum (mm)", value=25.0, min_value=0.0, max_value=600.0, step=1.0)
            dry_days = st.number_input("Dry days (consecutive &lt;1 mm)", value=0, min_value=0, max_value=60, step=1)
        temp_mean = (temp_max + temp_min) / 2
        inputs = {
            "location_id": float(loc_id),
            "temperature_2m_max": temp_max,
            "temperature_2m_min": temp_min,
            "temperature_mean": temp_mean,
            "precipitation_sum": precip,
            "precipitation_rolling_3d": roll_3d,
            "precipitation_rolling_7d": roll_7d,
            "dry_days": float(dry_days),
        }

    if st.button("Predict"):
        X = pd.DataFrame([inputs])  # ensure column order
        X = X[[c for c in feature_cols if c in X.columns]]
        proba = model.predict_proba(X)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        st.subheader("Prediction")
        st.metric("Extreme rain tomorrow?", "Yes (high risk)" if pred == 1 else "No (low risk)")
        st.metric("Estimated probability", f"{proba:.1%}")
        st.caption("Threshold 50 mm precipitation next day. For full SHAP explanations see `outputs/figures/shap_*.png`.")

    with st.expander("Feature importance (from SHAP)"):
        st.markdown("""
        - **Precipitation** (today and rolling 3d/7d) and **dry_days** are the main drivers.
        - **Temperature** and **location** add secondary signal.
        - Full plots: `outputs/figures/shap_summary.png`, `shap_importance.png`.
        """)


if __name__ == "__main__":
    main()
