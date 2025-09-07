import sys
print(">>> Python:", sys.version)

import numpy, pandas, sklearn
print(">>> numpy:", numpy.__version__)
print(">>> pandas:", pandas.__version__)
print(">>> scikit-learn:", sklearn.__version__)

import tensorflow as tf
print(">>> tensorflow:", tf.__version__)
print(">>> tf device list:", [d.device_type for d in tf.config.list_physical_devices()])

import qkeras, hls4ml
print(">>> qkeras:", qkeras.__version__)
print(">>> hls4ml:", hls4ml.__version__)

# quick sanity op
import numpy as np
a = np.random.randn(4, 3).astype("float32")
b = np.random.randn(3, 2).astype("float32")
print(">>> matmul ok:", (a @ b).shape)
