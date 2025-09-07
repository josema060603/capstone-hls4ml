# step1_get_data.py
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly

# ---------- Settings (you can change these) ----------
# Try JFK area as an example (you can set ATL, LAX, etc.)
lat, lon = 40.6413, -73.7781    # JFK
start = datetime(2020, 1, 1)
end   = datetime(2025, 1, 1)
out_path = "hourly_weather.csv"
# -----------------------------------------------------

print(f"Requesting hourly data for ({lat:.4f}, {lon:.4f}) from {start} to {end}")

pt = Point(lat, lon)

# First try: Observations only (model=False)
raw_obs = Hourly(pt, start, end, model=False).fetch()
print("Observations-only shape:", raw_obs.shape)
print("Obs columns:", list(raw_obs.columns))

# If observations are empty or 'temp' entirely NaN, fall back to model-backed
use_model = False
if raw_obs.empty or ('temp' in raw_obs and raw_obs['temp'].isna().all()):
    print("\nNo usable observations found. Falling back to model-backed data (reanalysis).")
    use_model = True
    raw = Hourly(pt, start, end, model=True).fetch()
else:
    raw = raw_obs

print("Selected dataset shape:", raw.shape)
if raw.empty:
    print("Still empty after model fallback. Try a different city or widen the dates.")
    raise SystemExit(0)

# Keep only columns that exist
keep_cols = [c for c in ['temp','dwpt','rhum','pres','wdir','wspd','prcp','snow'] if c in raw.columns]
if 'temp' not in keep_cols:
    print("No 'temp' column available even with model data. Try a different location.")
    raise SystemExit(0)

df = raw[keep_cols].copy()

# Clean to strict hourly grid and forward-fill small gaps
df = df[~df.index.duplicated(keep='first')].sort_index()
before = len(df)
df = df.asfreq('h')       # strict hourly index (lowercase 'h' to avoid FutureWarning)
df = df.ffill(limit=3)
df = df.dropna(subset=['temp'])
after = len(df)

print(f"Rows before clean: {before:,} | after clean: {after:,}")
print("Any NaN fraction per column:\n", df.isna().mean())

# Rename target and save
df = df.rename(columns={'temp': 'y'})
df.to_csv(out_path, index=True)
print(f"\nSaved {len(df):,} rows to {out_path}  (source: {'model' if use_model else 'observations'})")
print(df.head(3))
print(df.tail(3))
