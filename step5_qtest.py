import numpy as np
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, QActivation, quantized_bits, quantized_relu

MODEL_PATH = "tempnet_q8_b.h5"

print("TF:", tf.__version__, "Keras:", keras.__version__)

X_te = np.load("prepared/X_test.npy")
y_te = np.load("prepared/y_test.npy")

custom = {
    "QDense": QDense,
    "QActivation": QActivation,
    "quantized_bits": quantized_bits,
    "quantized_relu": quantized_relu,
}

# Load model (don’t use standalone keras.*; no need for safe_mode=True here)
model = keras.models.load_model(MODEL_PATH, custom_objects=custom, compile=False)

# Compile for evaluation
model.compile(loss="mse", metrics=["mae"])

# Evaluate
test_loss, test_mae = model.evaluate(X_te, y_te, verbose=0)
print(f"\nTest MAE: {test_mae:.3f} °C")

y_pred = model.predict(X_te, batch_size=1024, verbose=0).reshape(-1)
y_true = y_te.reshape(-1)   # or np.ravel(y_test)

err  = y_pred - y_true
mae  = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err**2)))
p90  = float(np.percentile(np.abs(err), 90))
p95  = float(np.percentile(np.abs(err), 95))

print("Shapes:", y_pred.shape, y_true.shape)
print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, p90={p90:.3f}, p95={p95:.3f}")
