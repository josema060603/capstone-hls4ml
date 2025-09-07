# step2_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from pathlib import Path

CSV_IN = "hourly_weather.csv"
OUT_DIR = Path("prepared")
OUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"Loading {CSV_IN} ...")
df = pd.read_csv(CSV_IN, parse_dates=[0], index_col=0)
df = df.sort_index()

# Ensure we have the target column 'y' (temp in °C), else try to map from 'temp'
if 'y' not in df.columns and 'temp' in df.columns:
    df = df.rename(columns={'temp':'y'})

assert 'y' in df.columns, "No 'y' column found (should be temperature in °C)."

# -----------------------------
# Feature engineering
# -----------------------------


H = 1  # predict next hour
LAGS = [1,2,3,6,12,24]

# Lags of the target
for l in LAGS:
    df[f'y_lag{l}'] = df['y'].shift(l) #creates a new column called y_lag{number of lags}  
    #which is equal to temperature column shifted down by the number of lags

# Rolling stats of the target
for w in [3,6,12,24]:
    df[f'y_rollmean_{w}h'] = df['y'].rolling(w).mean()# extracts the mean of the last time space and creates a new column
    df[f'y_rollstd_{w}h']  = df['y'].rolling(w).std() # extracts the standard deviation

# Optional exogenous features (keep only if present)
keep_exo = [c for c in ['dwpt','rhum','pres','wdir','wspd','prcp','wpgt','tsun','coco'] if c in df.columns]

# Cyclical time features
hour = df.index.hour
doy  = df.index.dayofyear
df['hour_sin'] = np.sin(2*np.pi*hour/24);   df['hour_cos'] = np.cos(2*np.pi*hour/24) #creates a new column with the hour coded in sin and cos
df['doy_sin']  = np.sin(2*np.pi*doy/365.25); df['doy_cos']  = np.cos(2*np.pi*doy/365.25) #creates a new column with the days coded in sin and cos

# Target at t+H
df['target'] = df['y'].shift(-H) #target is a shift 1 hour in the future




# Drop rows with any NaNs caused by shifting/rolling
features = (
    [f'y_lag{l}' for l in LAGS] +
    [f'y_rollmean_{w}h' for w in [3,6,12,24]] +
    [f'y_rollstd_{w}h'  for w in [3,6,12,24]] +
    keep_exo +
    ['hour_sin','hour_cos','doy_sin','doy_cos']
) 
# save what needs to be kept
required = (
    [f'y_lag{l}' for l in LAGS] +
    [f'y_rollmean_{w}h' for w in [3,6,12,24]] +
    [f'y_rollstd_{w}h'  for w in [3,6,12,24]] +
    ['target']
)
cols_to_keep = ['y','target'] + features

# keep all columns, but only drop rows missing in the required set
df = df[cols_to_keep].dropna(subset=required)

print("Final df shape:", df.shape)
print("Available dates:", df.index.min(), "to", df.index.max())

print("Final feature set:")
print(pd.Series(features).to_string(index=False))
print("\nDate range:", df.index.min(), "→", df.index.max())
print("Rows:", len(df), " | Features:", len(features))


# --- Fill occasional gaps in exogenous features ---
# Event-like (0 = no event):
for c in ['prcp','wpgt','tsun']:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

# Continuous weather signals: interpolate over time, then fill edges
for c in ['dwpt','rhum','pres','wspd','wdir']:
    if c in df.columns:
        df[c] = df[c].interpolate('time').ffill().bfill()

# Now proceed to define `required`, cols_to_keep, and drop only required NaNs as you already do
# required = [...lags/rolls..., 'target']
# df = df[cols_to_keep].dropna(subset=required)


# -----------------------------
# Time-based split
# -----------------------------
# Use last ~2 months for test, previous ~2 months for val (adjust as you like)
# If your dataset is long (~5 years), this yields plenty of train data
split1 = df.index.max() - pd.Timedelta(days=120)  # train/val boundary
split2 = df.index.max() - pd.Timedelta(days=60)   # val/test boundary

train = df.loc[:split1] # grab all data from beggining up to 120 days timestamp
val   = df.loc[split1:split2]#validation data from penultimate 60 days
test  = df.loc[split2:] #test data for last 60 days

# Avoid overlapping a single timestamp in two sets
train = train.iloc[:-1] if len(train) and len(val) else train

X_tr, y_tr = train[features].values, train['target'].values
X_va, y_va = val[features].values,   val['target'].values
X_te, y_te = test[features].values,  test['target'].values

print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")
print("Train Width:", X_tr.shape[1])
print("Val Width:", X_va.shape[1])
print("Test Width:", X_te.shape[1])



# -----------------------------
# Scale features (fit on train only)
# -----------------------------
sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr)
X_va_s = sc.transform(X_va)
X_te_s = sc.transform(X_te)

# -----------------------------
# Save prepared arrays and scaler params
# -----------------------------
np.save(OUT_DIR/"X_train.npy", X_tr_s)
np.save(OUT_DIR/"y_train.npy", y_tr)
np.save(OUT_DIR/"X_val.npy",   X_va_s)
np.save(OUT_DIR/"y_val.npy",   y_va)
np.save(OUT_DIR/"X_test.npy",  X_te_s)
np.save(OUT_DIR/"y_test.npy",  y_te)

# Save scaler parameters for use on PYNQ later
np.save(OUT_DIR/"scaler_mean.npy", sc.mean_.astype(np.float32))
np.save(OUT_DIR/"scaler_scale.npy", sc.scale_.astype(np.float32))
with open(OUT_DIR/"feature_names.txt","w") as f:
    f.write("\n".join(features))

print(f"\nSaved preprocessed data & scaler to: {OUT_DIR.resolve()}")
print("Example row (first train sample):")
print(pd.Series(X_tr_s[0], index=features).head(10))
