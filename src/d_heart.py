from ucimlrepo import fetch_ucirepo
from collections import Counter
import numpy as np
import pandas as pd

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Data (as pandas dataframes)
X = heart_disease.data.features.to_numpy()
y = heart_disease.data.targets.to_numpy()


# Task 1
def sigmoid(z):
  return 1 / (1 + np.exp(-z))


def logistic_regression_fit(X, y, learning_rate=0.01, num_iterations=1000):
  m, n = X.shape
  theta = np.zeros(
      (n + 1, 1))  # Initialize theta, adding one for the intercept term

  # Add a column of ones to X for the intercept term
  X = np.column_stack((np.ones(m), X))

  for _ in range(num_iterations):
    z = np.dot(X, theta)
    h = sigmoid(z)
    gradient = np.dot(X.T, (h - y)) / m
    theta -= learning_rate * gradient

  return theta


def logistic_regression_predict(X, theta):
  # Add a column of ones to X for the intercept term
  X = np.column_stack((np.ones(X.shape[0]), X))
  z = np.dot(X, theta)
  h = sigmoid(z)
  y_pred = (h >= 0.5).astype(int)
  return y_pred


# Task 2
def knn_predict(X_train, y_train, X, k=3):
  y_pred = [knn_predict_single(x, X_train, y_train, k) for x in X]
  return np.array(y_pred)


def knn_predict_single(x, X_train, y_train, k=3):
  distances = [np.linalg.norm(x - x_train) for x_train in X_train]
  k_indices = np.argsort(distances)[:k]
  k_nearest_labels = [tuple(y_train[i])
                      for i in k_indices]  # Convert y_train to a tuple
  most_common = Counter(k_nearest_labels).most_common(1)
  return most_common[0][0]


# Task 3
def evaluate_acc(y_true, y_pred):
  correct = np.sum(y_true == y_pred)
  accuracy = correct / len(y_true)
  return accuracy


# Testing
# Split data into training and testing sets (you can modify this as needed)
# For example, you can use the first 80% of the data for training and the remaining 20% for testing
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Test Logistic Regression
learning_rate = 0.01
num_iterations = 1000
theta = logistic_regression_fit(X_train, y_train, learning_rate,
                                num_iterations)
y_pred_logistic = logistic_regression_predict(X_test, theta)

# Test K-Nearest Neighbor
k = 3
y_pred_knn = knn_predict(X_train, y_train, X_test, k)

# Evaluation
accuracy_logistic = evaluate_acc(y_test, y_pred_logistic)
accuracy_knn = evaluate_acc(y_test, y_pred_knn)

print("Accuracy (Logistic Regression):", accuracy_logistic)
print("Accuracy (K-Nearest Neighbor):", accuracy_knn)
