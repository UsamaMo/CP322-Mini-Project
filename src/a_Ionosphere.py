from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Step 1: Fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Step 2: Convert data to NumPy arrays
X = ionosphere.data.features.to_numpy()
y = ionosphere.data.targets.to_numpy()

print("X array: ", X)
print("Y array: ", y)

# Step 3: Check for NaN values in X
# If this prints "Data in X contains NaN values", you would normally remove or impute these.
has_nan_X = np.isnan(X).any()
if has_nan_X:
    print("Data in X contains NaN values.")
    # Your cleaning code here

# Step 4: Label-encode y
# Useful for models that can interpret the ordinal relationship between labels
le = LabelEncoder()
y_encoded = le.fit_transform(y.ravel())  # Using ravel() to convert y to 1D array

# Step 5: One-hot encode y
# Useful for models that cannot interpret the ordinal relationship between labels
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y)

print("Encoded y:", y_encoded)
print("One-hot encoded y:", y_onehot)
