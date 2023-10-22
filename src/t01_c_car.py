from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt


#TASK 1

# Fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# Load data into NumPy arrays
X = car_evaluation.data.features.to_numpy()
y = car_evaluation.data.targets.to_numpy()

# Step 1: Load datasets into NumPy (Done)

# Step 2: Clean Data
# Check for Missing or Undefined Categories
missing_rows = []
for i in range(X.shape[0]):
    if None in X[i] or '' in X[i]:  # Check for None or empty strings as missing values
        missing_rows.append(i)

if missing_rows:
    print(f"Warning: Missing values found in rows: {missing_rows}")
    X = np.delete(X, missing_rows, axis=0)
    y = np.delete(y, missing_rows, axis=0)

# Encode the labels of the original features to integers
label_encoder_features = LabelEncoder()
X_integer_encoded = np.apply_along_axis(label_encoder_features.fit_transform, 0, X)

# One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Encode labels in y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.ravel())

# Step 3: Basic Statistics
# Class Distribution
(unique, counts) = np.unique(y_encoded, return_counts=True)
print("Class distribution: ", dict(zip(unique, counts)))

# Feature Stats (Now that X is numerical)
mean_features = np.mean(X_encoded, axis=0)
std_features = np.std(X_encoded, axis=0)
print("Mean: ", mean_features)
print("Std: ", std_features)

# Correlation (Now that X is numerical)
correlation_matrix = np.corrcoef(X_encoded, rowvar=False)
print("Correlation Matrix: ", correlation_matrix)

# Scatter Plots for multiple feature pairs
num_features = X_integer_encoded.shape[1]
plt.figure(figsize=(15,15))
plot_count = 1
for i in range(num_features):
    for j in range(i+1, num_features):
        plt.subplot(num_features, num_features, plot_count)
        plt.scatter(X_integer_encoded[:, i], X_integer_encoded[:, j], c=y_encoded, cmap='viridis', s=1)
        plt.xlabel(f'Feature {i+1}')
        plt.ylabel(f'Feature {j+1}')
        plot_count += 1
plt.tight_layout()
plt.show()


#TASK 2

# Logistic Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            predictions = 1 / (1 + np.exp(-model))

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = 1 / (1 + np.exp(-model))
        return np.round(predictions)

# k-Nearest Neighbor (KNN) Class
class KNN:
    def __init__(self, K=3):
        self.K = K
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.K]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        mode_result = mode(k_nearest_labels)
        if np.isscalar(mode_result.mode):
            most_common = mode_result.mode
        else:
            most_common = mode_result.mode[0]
        
        return most_common


def evaluate_acc(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Function to split the dataset into k-folds
def k_fold_split(X, y, k):
    fold_size = len(X) // k
    X_folds = []
    y_folds = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i != k - 1 else None
        X_folds.append(X[start_idx:end_idx])
        y_folds.append(y[start_idx:end_idx])
    return X_folds, y_folds

# Example of using the models and evaluating them:
k_folds = 5
X_folds, y_folds = k_fold_split(X_encoded, y_encoded, k_folds)

for k in range(1, 6):  # Searching for best K for KNN
    total_accuracy = 0
    for i in range(k_folds):
        X_train = np.vstack([fold for j, fold in enumerate(X_folds) if j != i])
        y_train = np.hstack([fold for j, fold in enumerate(y_folds) if j != i])
        X_val = X_folds[i]
        y_val = y_folds[i]

        # Using KNN as an example:
        knn = KNN(K=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        total_accuracy += evaluate_acc(y_val, y_pred)
    average_accuracy = total_accuracy / k_folds
    print(f"KNN K={k} Average Accuracy: {average_accuracy:.2f}")



    #TASK 3

# 1. Compare the accuracy of k-nearest neighbor and logistic regression on the four datasets.
knn_accuracies = []
log_reg_accuracies = []

for i in range(k_folds):
    X_train = np.vstack([fold for j, fold in enumerate(X_folds) if j != i])
    y_train = np.hstack([fold for j, fold in enumerate(y_folds) if j != i])
    X_val = X_folds[i]
    y_val = y_folds[i]

    # KNN
    knn = KNN(K=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_val)
    knn_accuracies.append(evaluate_acc(y_val, y_pred_knn))
    
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_val)
    log_reg_accuracies.append(evaluate_acc(y_val, y_pred_log_reg))

print(f"Average KNN Accuracy: {np.mean(knn_accuracies):.2f}")
print(f"Average Logistic Regression Accuracy: {np.mean(log_reg_accuracies):.2f}")

# 2. Test different k values for KNN
k_values = list(range(1, 16))
knn_k_accuracies = []

for k in k_values:
    knn = KNN(K=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    knn_k_accuracies.append(evaluate_acc(y_val, y_pred))

plt.plot(k_values, knn_k_accuracies)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy for KNN")
plt.show()

# 3. Test different learning rates for Logistic Regression
learning_rates = [0.1, 0.01, 0.001, 0.0001]
log_reg_accuracies_lr = []

for lr in learning_rates:
    log_reg = LogisticRegression(learning_rate=lr)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_val)
    log_reg_accuracies_lr.append(evaluate_acc(y_val, y_pred))

plt.plot(learning_rates, log_reg_accuracies_lr)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Learning Rate vs Accuracy for Logistic Regression")
plt.show()

# 4. Compare accuracy of models as a function of dataset size
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
knn_accuracies_size = []
log_reg_accuracies_size = []

for size in train_sizes:
    end_idx = int(len(X_encoded) * size)
    X_subset = X_encoded[:end_idx]
    y_subset = y_encoded[:end_idx]

    # KNN
    knn = KNN(K=3)
    knn.fit(X_subset, y_subset)
    y_pred_knn = knn.predict(X_val)
    knn_accuracies_size.append(evaluate_acc(y_val, y_pred_knn))
    
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_subset, y_subset)
    y_pred_log_reg = log_reg.predict(X_val)
    log_reg_accuracies_size.append(evaluate_acc(y_val, y_pred_log_reg))

plt.plot(train_sizes, knn_accuracies_size, label="KNN")
plt.plot(train_sizes, log_reg_accuracies_size, label="Logistic Regression")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Training Size vs Accuracy")
plt.legend()
plt.show()

