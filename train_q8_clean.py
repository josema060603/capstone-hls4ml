#!/usr/bin/env python3
"""
train_q8_clean.py
Clean retrain script (TF-Keras only, no Lambda/bias_add).

- Topology: 64 -> 32 -> 1
- Quantizers:
    qdense1.kernel: quantized_bits(8,3,1)   (signed)
    qact1:          quantized_relu(8,4)     (unsigned)
    qdense2.kernel: quantized_bits(8,2,1)   (signed)
- Biases stay float32 in training (quantize/widen later in HLS).
- Saves: full H5, SavedModel dir, and weights H5.
"""

import os, random, numpy as np
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoid oneDNN drift

SEED = 42
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

# ---- Load data (expects prepared/*.npy). Fallback to toy data if not present. ----
def load_or_dummy():
    import os
    if all(os.path.exists(p) for p in [
        "prepared/X_train.npy","prepared/y_train.npy",
        "prepared/X_val.npy","prepared/y_val.npy",
        "prepared/X_test.npy","prepared/y_test.npy"
    ]):
        X_tr = np.load("prepared/X_train.npy")
        y_tr = np.load("prepared/y_train.npy")
        X_va = np.load("prepared/X_val.npy")
        y_va = np.load("prepared/y_val.npy")
        X_te = np.load("prepared/X_test.npy")
        y_te = np.load("prepared/y_test.npy")
    else:
        Ntr, Nva, Nte, FEAT = 4000, 500, 500, 64
        X_tr = np.random.randn(Ntr, FEAT).astype("float32")
        y_tr = (X_tr[:, :4].sum(axis=1, keepdims=True) * 0.1).astype("float32")
        X_va = np.random.randn(Nva, FEAT).astype("float32")
        y_va = (X_va[:, :4].sum(axis=1, keepdims=True) * 0.1).astype("float32")
        X_te = np.random.randn(Nte, FEAT).astype("float32")
        y_te = (X_te[:, :4].sum(axis=1, keepdims=True) * 0.1).astype("float32")
    return X_tr, y_tr, X_va, y_va, X_te, y_te

X_tr, y_tr, X_va, y_va, X_te, y_te = load_or_dummy()
assert X_tr.shape[1] == 64, f"Expected 64 features, got {X_tr.shape[1]}"

# ---- Build model (NO Lambda layers) ----
inp = keras.layers.Input((64,), name="input")
x   = QDense(32, kernel_quantizer=quantized_bits(8,3,1), bias_quantizer=None, use_bias=True, name="qdense1")(inp)
x   = QActivation(quantized_relu(8,4), name="qact1")(x)
out = QDense(1,  kernel_quantizer=quantized_bits(8,2,1), bias_quantizer=None, use_bias=True, name="qdense2")(x)
out = keras.layers.Identity(name="qout")(out)  # keep named endpoint

model = keras.Model(inp, out, name="tempnet_q8")
model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])

cbs = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
]

model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=100,
    batch_size=64,
    shuffle=False,
    callbacks=cbs,
    verbose=2,
)

test_loss, test_mae = model.evaluate(X_te, y_te, verbose=0)
print(f"[Q8 clean] Test MAE: {test_mae:.4f}")

# ---- Save formats that are robust for hls4ml ----
# 1) Legacy full H5 (easy reload with tf.keras + custom_objects)
model.save("tempnet_q8_full.h5")
# 2) SavedModel directory
tf.saved_model.save(model, "tempnet_q8_savedmodel")
# 3) Weights (optional)
model.save_weights("tempnet_q8.weights.h5")
print("Saved: tempnet_q8_full.h5, tempnet_q8_savedmodel/, tempnet_q8.weights.h5")
