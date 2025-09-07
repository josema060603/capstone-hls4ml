import numpy as np
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
import hls4ml

# --- load your test set (same preprocessing as training) ---
X_test = np.load("prepared/X_test.npy").astype("float32")
y_test = np.load("prepared/y_test.npy").astype("float32")

# --- load the trained QAT model you saved ---
model = keras.models.load_model(
    "tempnet_q8_b.h5",
    custom_objects={
        "QDense": QDense, "QActivation": QActivation,
        "quantized_bits": quantized_bits, "quantized_relu": quantized_relu
    },
    compile=False
)
# --- put this AFTER you've loaded `model`, `X_test`, `y_test` ---
import os, math, numpy as np
from tensorflow import keras
from qkeras import QDense
import hls4ml

# 1) Start from auto config
cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
cfg['Model']['ReuseFactor'] = 32  # default for any layer not overridden
cfg['LayerName']['qdense1']['ReuseFactor'] = 32   # 1536/32 ≈ 48 parallel mult
cfg['LayerName']['qdense2']['ReuseFactor'] = 64   # 2048/64 ≈ 32 parallel mult

# 2) IO tensor names
in_name  = model.input.name.split(':')[0]
out_name = model.output.name.split(':')[0]

# 3) Measure ranges (for sane fixed-point choices)
X_test = X_test.astype('float32')
y_true = y_test.reshape(-1).astype('float32')
y_pred_keras = model.predict(X_test, verbose=0).reshape(-1)
x_max = float(np.max(np.abs(X_test))) + 1e-9
y_max = float(np.max(np.abs(y_pred_keras))) + 1e-9
print("x_max:", x_max, "y_max:", y_max)

# 4) Widen INPUT; keep OUTPUT reasonable (±32 is fine for y_max≈24)
cfg['LayerName'].setdefault(in_name, {}).setdefault('Precision', {})
cfg['LayerName'][in_name]['Precision']['result'] = 'ap_fixed<24,12>'

cfg['LayerName'].setdefault(out_name, {}).setdefault('Precision', {})
cfg['LayerName'][out_name]['Precision']['result'] = 'ap_fixed<16,6>'
# If you want a one-time sanity: use float and see MAE snap back
# cfg['LayerName'][out_name]['Precision']['result'] = 'float'

# 5) Prevent internal overflow: widen Dense accumulators + pre-activation result
dense_names = [l.name for l in model.layers
               if isinstance(l, QDense) or isinstance(l, keras.layers.Dense)]
for n in dense_names:
    cfg['LayerName'].setdefault(n, {}).setdefault('Precision', {})
    cfg['LayerName'][n]['Precision']['accum']  = 'ap_fixed<24,12>'
    cfg['LayerName'][n]['Precision']['result'] = 'ap_fixed<24,12>'

# 6) Convert, emulate, and check MAE
outdir = os.path.expanduser('~/capstone2/hls4ml_prj')
os.makedirs(outdir, exist_ok=True)

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    backend='Vitis',
    part='xc7z020clg400-1',
    clock_period=10,
    io_type='io_parallel',      # or 'io_stream' if you prefer
    output_dir=outdir
)

hls_model.compile()
y_hls = hls_model.predict(np.ascontiguousarray(X_test)).reshape(-1)
mae = float(np.mean(np.abs(y_hls - y_true)))
print("HLS emu MAE:", mae)
print("y_hls min/max:", float(y_hls.min()), float(y_hls.max()))
