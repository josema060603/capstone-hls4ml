import numpy as np

# Load arrays
X_tr = np.load("prepared/X_train.npy")
y_tr = np.load("prepared/y_train.npy")

print("X_tr shape:", X_tr.shape)
print("y_tr shape:", y_tr.shape)

# Print first 5 rows of features
print("First 5 rows of X_tr:\n", X_tr[100:105])

# Print first 5 targets
print("First 5 targets of y_tr:\n", y_tr[100:105])
