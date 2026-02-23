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
    st.title("🌧️ Extreme Rain Tomorrow : Sri Lanka Agriculture")
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
        row_idx = st.sidebar.slider("Row index (test set)", 0, len(test_df) - 1, 0)
        sample = test_df.iloc[row_idx]
        st.sidebar.write("Selected row date:", sample.get("date", "—"))
        inputs = {c: float(sample[c]) for c in feature_cols if c in sample.index}
    else:
        # Manual inputs — optional presets for demo (low / high risk)
        preset = st.sidebar.selectbox(
            "Preset (manual input)",
            ["None", "Demo: Low risk", "Demo: High risk"],
            index=0,
            help="Pre-fill values for a clear low- or high-risk prediction.",
        )
        if preset == "Demo: High risk":
            # Real heavy-rain day from test set (2024-11-25 Batticaloa): ensures model gives elevated probability
            def_loc, def_tmax, def_tmin = 1, 26.0, 23.8
            def_precip, def_3d, def_7d, def_dry = 101.8, 143.4, 209.8, 0
        elif preset == "Demo: Low risk":
            def_loc, def_tmax, def_tmin = 0, 30.0, 24.0
            def_precip, def_3d, def_7d, def_dry = 1.0, 3.0, 8.0, 5
        else:
            def_loc, def_tmax, def_tmin = 0, 30.0, 24.0
            def_precip, def_3d, def_7d, def_dry = 5.0, 10.0, 25.0, 0

        st.subheader("Weather features (today)")
        loc_id = st.selectbox("Location", options=[0, 1, 2], format_func=lambda x: LOCATION_NAMES[x], index=int(def_loc))
        col1, col2 = st.columns(2)
        with col1:
            temp_max = st.number_input("Max temperature (°C)", value=def_tmax, min_value=15.0, max_value=45.0, step=0.5)
            temp_min = st.number_input("Min temperature (°C)", value=def_tmin, min_value=15.0, max_value=40.0, step=0.5)
            precip = st.number_input("Precipitation today (mm)", value=def_precip, min_value=0.0, max_value=200.0, step=0.5)
            roll_3d = st.number_input("Precipitation 3-day sum (mm)", value=def_3d, min_value=0.0, max_value=400.0, step=1.0)
        with col2:
            roll_7d = st.number_input("Precipitation 7-day sum (mm)", value=def_7d, min_value=0.0, max_value=600.0, step=1.0)
            dry_days = st.number_input("Dry days (consecutive &lt;1 mm)", value=def_dry, min_value=0, max_value=60, step=1)
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

    # Model is very conservative (~0.7% for heavy-rain, ~0.01% for low-rain). Use 0.5% so elevated risk shows "High risk".
    RISK_THRESHOLD = 0.005

    if st.button("Predict"):
        # Ensure same column order and types as training (critical for XGBoost)
        X = pd.DataFrame([[float(inputs[c]) for c in feature_cols]], columns=feature_cols)
        proba = model.predict_proba(X)[0, 1]
        pred = 1 if proba >= RISK_THRESHOLD else 0
        st.subheader("Prediction")
        st.metric("Extreme rain tomorrow?", "Yes (high risk)" if pred == 1 else "No (low risk)")
        st.metric("Estimated probability", f"{proba:.1%}")
        st.caption(f"Risk threshold = {RISK_THRESHOLD*100:.1f}% (early-warning/demo). Target: next-day precip >50 mm. SHAP: `outputs/figures/shap_*.png`.")

    with st.expander("Feature importance (from SHAP)"):
        st.markdown("""
        - **Precipitation** (today and rolling 3d/7d) and **dry_days** are the main drivers.
        - **Temperature** and **location** add secondary signal.
        - Full plots: `outputs/figures/shap_summary.png`, `shap_importance.png`.
        """)


if __name__ == "__main__":
    main()
