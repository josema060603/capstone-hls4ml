import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# ---- Load model (.keras preferred; falls back to .h5) ----
keras_path = Path("tempnet_fp32.keras")
h5_path    = Path("tempnet_fp32.h5")

if keras_path.exists():
    model_path = keras_path
elif h5_path.exists():
    model_path = h5_path
    try:
        import h5py  # required for .h5
    except ImportError as e:
        raise ImportError("h5py is required to load .h5 models. pip install h5py") from e
else:
    raise FileNotFoundError("Couldn't find tempnet_fp32.keras or tempnet_fp32.h5 in this folder.")

model = keras.models.load_model(model_path, compile=False)
print(f"Loaded model: {model_path.name}")

# ---- Load data & predict ----
X_te = np.load("prepared/X_test.npy")
y_te = np.load("prepared/y_test.npy")
y_pred = model.predict(X_te, verbose=0).reshape(-1)

mae = mean_absolute_error(y_te, y_pred)
print(f"Test MAE: {mae:.3f} °C")

# ==== OPTIONS for how much to plot on the timeseries ====
# 1) Show the first N points (readable slice)
N = 200  # change to any number, or set to len(y_te) to plot all
start = 0             # start index of the slice
end   = min(start+N, len(y_te))

# 2) To plot ALL points, uncomment:
# start, end = 0, len(y_te)

# 3) To plot a different window, set start to an index and keep N length
# start = 500
# end   = min(start+N, len(y_te))

# ---- Timeseries: Pred vs Actual (slice) ----
plt.figure(figsize=(12,4))
plt.plot(y_te[start:end], label="Actual")
plt.plot(y_pred[start:end], "--", label="Predicted")
plt.xlabel("Test index")
plt.ylabel("Temperature (°C)")
plt.title(f"Pred vs Actual (test[{start}:{end}])   |   MAE={mae:.3f} °C")
plt.legend(); plt.grid(True); plt.tight_layout()

# If you prefer saving instead of showing:
# plt.savefig("pred_vs_actual_timeseries.png", dpi=150); plt.close()
plt.savefig("test1.png")


# ---- Scatter: Actual vs Predicted (all test points) ----
lo = float(min(y_te.min(), y_pred.min()))
hi = float(max(y_te.max(), y_pred.max()))
plt.figure(figsize=(6,6))
plt.scatter(y_te, y_pred, alpha=0.3, s=10)
plt.plot([lo, hi], [lo, hi], "--")  # perfect-fit line
plt.xlabel("Actual (°C)")
plt.ylabel("Predicted (°C)")
plt.title("Scatter: Actual vs Predicted (Test)")
plt.grid(True); plt.tight_layout()

# If you prefer saving:
# plt.savefig("scatter_actual_vs_pred.png", dpi=150); plt.close()
plt.savefig("test2.png")
