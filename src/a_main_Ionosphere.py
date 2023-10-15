from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Fetch and prepare data
ionosphere = fetch_ucirepo(id=52)
X = ionosphere.data.features.to_numpy()
y = ionosphere.data.targets.to_numpy()

# Basic Data Cleaning
has_nan_X = np.isnan(X).any()
if has_nan_X:
    print("Data in X contains NaN values.")
    # Your cleaning code here

# Label-encoding and one-hot encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y.ravel())
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
df['Class'] = y.ravel()

# Basic statistics
print("Basic Statistics:")
print(df.describe())

# Class Distribution
print("\nClass Distribution:")
print(df['Class'].value_counts())

# Feature Correlations
print("\nFeature Correlations:")
correlations = df.corr()
print(correlations)

# Plots
# Class Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

# Pairwise Scatter Plots
sns.pairplot(df, hue='Class')
plt.suptitle('Pairwise Scatter Plots of Features', y=1.02)
plt.show()

# Heatmap for Feature Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()
