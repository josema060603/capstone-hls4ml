import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from qkeras import QDense, QActivation
print("TF:", tf.__version__, "Keras:", keras.__version__)

X_tr = np.load("prepared/X_train.npy")
y_tr = np.load("prepared/y_train.npy")
X_va = np.load("prepared/X_val.npy")
y_va = np.load("prepared/y_val.npy")
X_te = np.load("prepared/X_test.npy")
y_te = np.load("prepared/y_test.npy")
n_features = X_tr.shape[1]

wq1 = "quantized_bits(8, 3, 1)"   # 8-bit weights, 3 integer bits, signed
wq2 = "quantized_bits(8, 2, 1)"
bq  = "quantized_bits(8, 3, 1)"   # quantized bias prevents bias_add JSON issues
aq  = "quantized_relu(8, 4)"      # 8-bit ReLU activations

# --- Model ---
inp = keras.Input(shape=(n_features,), name="in_x")
x   = QDense(64, kernel_quantizer=wq1, bias_quantizer=bq, name="qdense1")(inp)
x   = QActivation(aq, name="qact1")(x)
x   = QDense(32, kernel_quantizer=wq2, bias_quantizer=bq, name="qdense2")(x)
x   = QActivation(aq, name="qact2")(x)

# Keep output float for best regression accuracy (quantize later if needed)
out = keras.layers.Dense(1, name="y")(x)
qmodel = keras.Model(inp, out, name="tempnet_q8")
qmodel.summary()

# --- Train (QAT) ---
qmodel.compile(
    optimizer=keras.optimizers.Adam(1e-4),   # QAT likes a smaller LR
    loss="mse",
    metrics=["mae"]
)

history = qmodel.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=150,
    batch_size=256,
    verbose=1
)

print("Best val MAE:", float(np.min(history.history["val_mae"])))

# --- Save (both formats work on TF 2.12) ---
qmodel.save("tempnet_q8_savedmodel_b")  # SavedModel directory
qmodel.save("tempnet_q8_b.h5")          # HDF5 file