from __future__ import annotations
import io
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "household_power_consumption.txt"
DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
FEATURES = ["lag_1", "lag_2", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28", "day_of_week", "month", "is_weekend"]

def ensure_dataset(raw_file: Path = RAW_FILE) -> Path:
    """Download and extract the official UCI archive if the raw file is absent."""
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    if raw_file.exists():
        return raw_file
    try:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "WattWiseAI/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            candidates = [n for n in archive.namelist() if n.lower().endswith("household_power_consumption.txt")]
            if not candidates:
                raise RuntimeError("The UCI archive did not contain the expected TXT file.")
            raw_file.write_bytes(archive.read(candidates[0]))
    except Exception as exc:
        raise RuntimeError(f"Could not download the UCI dataset. Check internet or place household_power_consumption.txt at {raw_file}. Original error: {exc}") from exc
    return raw_file

def load_daily_consumption(raw_file: Path = RAW_FILE) -> pd.Series:
    """Read source data and return daily mean active power in kW."""
    raw_file = ensure_dataset(raw_file)
    frame = pd.read_csv(raw_file, sep=";", na_values=["?"], usecols=["Date", "Time", "Global_active_power"], dtype={"Date":"string","Time":"string","Global_active_power":"float32"}, low_memory=False)
    timestamps = pd.to_datetime(frame["Date"] + " " + frame["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    frame["timestamp"] = timestamps
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    daily = frame["Global_active_power"].resample("D").mean().astype("float64")
    daily.name = "daily_mean_active_power_kw"
    daily = daily.interpolate(method="time", limit_direction="both")
    if daily.isna().any() or len(daily) < 60:
        raise ValueError("Insufficient valid daily data after cleaning.")
    return daily

def make_supervised(daily: pd.Series) -> pd.DataFrame:
    """Create one-day-ahead rows; predictors use only days before target day."""
    df = pd.DataFrame({"target": daily})
    for lag in [1, 2, 7, 14, 28]:
        df[f"lag_{lag}"] = daily.shift(lag)
    prior = daily.shift(1)
    df["rolling_mean_7"] = prior.rolling(7, min_periods=7).mean()
    df["rolling_mean_28"] = prior.rolling(28, min_periods=28).mean()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df.dropna(subset=["target"] + FEATURES)

def chronological_split(data: pd.DataFrame, train_fraction: float = 0.8):
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1).")
    split_at = int(len(data) * train_fraction)
    if split_at <= 0 or split_at >= len(data):
        raise ValueError("Not enough rows for the requested split.")
    return data.iloc[:split_at].copy(), data.iloc[split_at:].copy()

def make_next_day_features(daily: pd.Series) -> pd.DataFrame:
    """Build feature row for the day immediately after the latest reading."""
    if len(daily) < 28:
        raise ValueError("At least 28 daily observations are required.")
    next_date = daily.index.max() + pd.Timedelta(days=1)
    row = {"lag_1":float(daily.iloc[-1]),"lag_2":float(daily.iloc[-2]),"lag_7":float(daily.iloc[-7]),"lag_14":float(daily.iloc[-14]),"lag_28":float(daily.iloc[-28]),"rolling_mean_7":float(daily.iloc[-7:].mean()),"rolling_mean_28":float(daily.iloc[-28:].mean()),"day_of_week":int(next_date.dayofweek),"month":int(next_date.month),"is_weekend":int(next_date.dayofweek >= 5)}
    return pd.DataFrame([row], index=[next_date], columns=FEATURES)
