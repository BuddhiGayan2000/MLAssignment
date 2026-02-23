import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
)

# Threshold (mm) for "extreme rain" / high flood-risk day
EXTREME_RAIN_MM = 50.0
# Rolling windows (days) for antecedent precipitation
ROLLING_DAYS = [3, 7]


def load_raw() -> pd.DataFrame:
    """Load raw weather CSV."""
    path = RAW_DIR / "weather_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run: python -m src.fetch_weather"
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling precipitation, mean temp, and dry-spell proxy per location."""
    out = []
    for loc, g in df.groupby("location", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        # Rolling precipitation (antecedent)
        for d in ROLLING_DAYS:
            g[f"precipitation_rolling_{d}d"] = g["precipitation_sum"].rolling(d, min_periods=1).sum()
        g["temperature_mean"] = (g["temperature_2m_max"] + g["temperature_2m_min"]) / 2
        # Dry spell: consecutive days with precip < 1 mm (simplified: days since last rain > 5mm)
        precip = g["precipitation_sum"].fillna(0)
        g["dry_days"] = (precip < 1).astype(int).groupby((precip >= 1).cumsum()).cumcount()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target: extreme_rain_tomorrow = 1 if next day's precipitation_sum > EXTREME_RAIN_MM (forecast)."""
    df = df.copy()
    df["extreme_rain_day"] = (df["precipitation_sum"] > EXTREME_RAIN_MM).astype(int)
    # Per-location shift so we predict tomorrow from today
    df["extreme_rain_tomorrow"] = df.groupby("location")["extreme_rain_day"].shift(-1)
    return df


def drop_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or inf in key columns or target."""
    key_cols = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "precipitation_rolling_3d", "precipitation_rolling_7d", "temperature_mean", "dry_days",
        "extreme_rain_tomorrow",
    ]
    for c in key_cols:
        if c in df.columns:
            df = df[np.isfinite(df[c].replace([np.inf, -np.inf], np.nan)) & df[c].notna()]
    return df.reset_index(drop=True)


def train_val_test_split_by_time(df: pd.DataFrame):
    """Split by date (no future leakage): first TRAIN_SIZE, then VAL_SIZE, then TEST_SIZE."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    t1 = int(n * TRAIN_SIZE)
    t2 = int(n * (TRAIN_SIZE + VAL_SIZE))
    train = df.iloc[:t1]
    val = df.iloc[t1:t2]
    test = df.iloc[t2:]
    return train, val, test


def get_feature_columns() -> list:
    """Numeric feature columns used for modeling (exclude identifiers and target)."""
    return [
        "temperature_2m_max", "temperature_2m_min", "temperature_mean",
        "precipitation_sum", "precipitation_rolling_3d", "precipitation_rolling_7d",
        "dry_days",
    ]


def main():
    df = load_raw()
    df = add_derived_features(df)
    df = add_target(df)
    df = drop_missing_and_infinite(df)

    # Encode location for optional use (tree models can use numeric location id)
    loc_map = {name: i for i, name in enumerate(df["location"].unique())}
    df["location_id"] = df["location"].map(loc_map)
    feature_cols = get_feature_columns()
    if "location_id" not in feature_cols:
        feature_cols = ["location_id"] + feature_cols

    train, val, test = train_val_test_split_by_time(df)

    # Save full processed table and splits
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "weather_processed.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Save feature list for train/evaluate/explain
    with open(PROCESSED_DIR / "feature_columns.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"Processed {len(df)} rows; train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Target 'extreme_rain_tomorrow' prevalence: train={train['extreme_rain_tomorrow'].mean():.3f}")
    return df, train, val, test


if __name__ == "__main__":
    main()
