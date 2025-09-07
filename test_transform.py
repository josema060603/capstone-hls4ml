import numpy as np
from sklearn.preprocessing import StandardScaler

# pretend dataset: 3 rows × 2 features
X = np.array([
    [50, 1000],   # row 1
    [60, 1010],   # row 2
    [70, 1020]    # row 3
])

print("Original X:\n", X)

# 1. Fit the scaler on X
sc = StandardScaler().fit(X)

print("\nStored means:", sc.mean_)     # one mean per column
print("Stored stds:", sc.scale_)       # one std per column

# 2. Transform the data
X_scaled = sc.transform(X)

print("\nScaled X:\n", X_scaled)

# 3. Check: after scaling, each column should have mean≈0, std≈1
print("\nColumn means after scaling:", X_scaled.mean(axis=0))
print("Column stds after scaling: ", X_scaled.std(axis=0))
