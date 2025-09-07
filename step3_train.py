# step3_train.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

DATA_DIR = Path("prepared")

# ---------------------------
# 1. Load prepared data
# ---------------------------
X_tr = np.load(DATA_DIR/"X_train.npy")
y_tr = np.load(DATA_DIR/"y_train.npy")
X_va = np.load(DATA_DIR/"X_val.npy")
y_va = np.load(DATA_DIR/"y_val.npy")
X_te = np.load(DATA_DIR/"X_test.npy")
y_te = np.load(DATA_DIR/"y_test.npy")

print("Shapes:", X_tr.shape, y_tr.shape, X_va.shape, y_va.shape, X_te.shape, y_te.shape)

n_features = X_tr.shape[1]

# ---------------------------
# 2. Build simple neural net
# ---------------------------
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)   # regression: predict next-hour temp
])
print("Check X_tr:", np.isnan(X_tr).sum(), "NaNs,", np.isinf(X_tr).sum(), "infs")
print("Check y_tr:", np.isnan(y_tr).sum(), "NaNs,", np.isinf(y_tr).sum(), "infs")
print("X_tr min/max:", np.min(X_tr), np.max(X_tr))
print("y_tr min/max:", np.min(y_tr), np.max(y_tr))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ---------------------------
# 3. Train
# ---------------------------
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=20,
    batch_size=128,
    verbose=2
)

# ---------------------------
# 4. Evaluate
# ---------------------------
test_loss, test_mae = model.evaluate(X_te, y_te, verbose=0)
print(f"\nTest MAE: {test_mae:.3f} °C")

# ---------------------------
# 5. Save model
# ---------------------------
model.save("tempnet_fp32.keras")
print("Saved model to tempnet_fp32.keras")


# # Save predictions + training history for plotting without TF
# import json, numpy as np
# y_pred = model.predict(X_te, verbose=0).reshape(-1)
# np.save("y_pred_test.npy", y_pred)
# with open("training_history.json", "w") as f:
#     json.dump(history.history, f)
# print("Saved y_pred_test.npy and training_history.json")
