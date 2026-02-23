import requests
import pandas as pd
from pathlib import Path

from src.config import RAW_DIR

# Sri Lanka: agricultural hubs (Central, Eastern, North Central)
LOCATIONS = [
    {"name": "Kandy", "lat": 7.2931, "lon": 80.6333},   # Central Province
    {"name": "Batticaloa", "lat": 7.7167, "lon": 81.7000},  # Eastern Province
    {"name": "Anuradhapura", "lat": 8.3350, "lon": 80.4100},  # North Central (paddy)
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


def fetch_location(lat: float, lon: float, start: str, end: str) -> dict:
    """Request daily weather for one location. Returns API response as dict."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Colombo",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    rows = []
    for loc in LOCATIONS:
        data = fetch_location(loc["lat"], loc["lon"], START_DATE, END_DATE)
        daily = data.get("daily", {})
        times = daily.get("time", [])
        n = len(times)
        for i in range(n):
            rows.append({
                "date": times[i],
                "location": loc["name"],
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "temperature_2m_max": daily.get("temperature_2m_max", [None] * n)[i],
                "temperature_2m_min": daily.get("temperature_2m_min", [None] * n)[i],
                "precipitation_sum": daily.get("precipitation_sum", [None] * n)[i],
            })
    df = pd.DataFrame(rows)
    out = RAW_DIR / "weather_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return df


if __name__ == "__main__":
    main()
