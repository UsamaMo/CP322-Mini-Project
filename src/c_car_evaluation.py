from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------
#TASK 1
#-------------------------------------------------------------------------------------------------------------------

# Fetch the dataset from https://archive.ics.uci.edu/dataset/19/car+evaluation
car_evaluation = fetch_ucirepo(id=19)

# Step 1: Loading the data into NumPy arrays
X = car_evaluation.data.features.to_numpy()
y = car_evaluation.data.targets.to_numpy()


# Step 2: Cleaning the Data
# Checking for Missing or Undefined Categories

missing_rows = []
for i in range(X.shape[0]):
    if None in X[i] or '' in X[i]:  #Checking if there is None in X array or empty space
        missing_rows.append(i)      #then we add those empty spaces to missing rows


#If there are missing rows found, then delete the missing rows
if missing_rows:
    X = np.delete(X, missing_rows, axis=0)
    y = np.delete(y, missing_rows, axis=0)


# Encode Labels for X
label_encoder = LabelEncoder()  #creates a label encoder object
X_integer_encoded = np.apply_along_axis(label_encoder.fit_transform, 0, X) #convert categories in X to integers

# One-Hot Encoding
encoder = OneHotEncoder(sparse=False) #Creates a one hot encoder object
X_encoded = encoder.fit_transform(X) #This applies one hot encoding to X

# Encode labels for y
label_encoder = LabelEncoder() #creates another label encoder object
y_encoded = label_encoder.fit_transform(y.ravel())

# Step 3: Basic Statistics

#compute the mean standard deviation
mean_features = np.mean(X_encoded)
std_features = np.std(X_encoded)

print("Mean: ", mean_features)                                      #Prints the Mean
print("Std: ", std_features)                                        #Prints the Standard Deviation
                                                
#-------------------------------------------------------------------------------------------------------------------
#TASK 2
#-------------------------------------------------------------------------------------------------------------------

# Logistic Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.001, iterations=1000):   #Initializes logistic regression model
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):                                        #Trains the logistic regression model            
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = 1 / (1 + np.exp(-model))

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):                                       #Makes prediction when given new data sets for the logistic regression model
        model = np.dot(X, self.weights) + self.bias
        logistic_predictions = 1 / (1 + np.exp(-model))
        return np.round(logistic_predictions)

# k-Nearest Neighbor (KNN) Class
class KNN:                      
    def __init__(self, K=3):                                    #Initializes the KNN model
        self.K = K
    
    def fit(self, X, y):                                        #Trains the KNN model
        self.X_train = X
        self.y_train = y

    def predict(self, X):                                       #Makes prediction when given new data sets for the KNN model
        knn_predictions = [self._predict(x) for x in X]
        return np.array(knn_predictions)
    
    def _predict(self, x):                                                          #Helper function for helping predict new data
        distances = [np.linalg.norm(x - training_samples) for training_samples in self.X_train]       #Calculates the euclidean distance between x and all training samples
        k_indices = np.argsort(distances)[:self.K]                                                    #sorts the distance  with smallest K indices
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        mode_result = mode(k_nearest_labels)
        if np.isscalar(mode_result.mode):
            most_common = mode_result.mode
        else:
            most_common = mode_result.mode[0]
        
        return most_common


def accuracy(true_y, predicted_y):                           #calculates the accuracy by comparing true y and predicted y values
    accuracy = np.sum(true_y == predicted_y) / len(true_y)
    return accuracy

# K fold cross validation function
def k_fold_split(X, y, k):
    fold_size = len(X) // k
    X_fold = []
    y_fold = []
    for i in range(k):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size if i != k - 1 else None
        X_fold.append(X[start_index:end_index])
        y_fold.append(y[start_index:end_index])
    return X_fold, y_fold

#Here we are setting the k fold cross validation value to 5
k_folds = 5                                                                             
X_folds, y_folds = k_fold_split(X_encoded, y_encoded, k_folds)

for k in range(1, 6):  # Searching for best K for KNN
    total_accuracy = 0
    for i in range(k_folds):
        X_train = np.vstack([fold for j, fold in enumerate(X_folds) if j != i])
        y_train = np.hstack([fold for j, fold in enumerate(y_folds) if j != i])
        X_val = X_folds[i]
        y_tested = y_folds[i]

        # Using KNN as an example:
        knn = KNN(K=k)
        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_val)
        total_accuracy += accuracy(y_tested, y_predicted)
    average_accuracy = total_accuracy / k_folds
    print(f"KNN K={k} Average Accuracy: {average_accuracy:.2f}")



#-------------------------------------------------------------------------------------------------------------------
#TASK 3
#-------------------------------------------------------------------------------------------------------------------

# 1. Compare the accuracy of k-nearest neighbor and logistic regression on the four datasets.
accuracy_knn = []
accuracy_logistic = []

for i in range(k_folds):
    X_train = np.vstack([fold for j, fold in enumerate(X_folds) if j != i])
    y_train = np.hstack([fold for j, fold in enumerate(y_folds) if j != i])
    X_val = X_folds[i]
    y_tested = y_folds[i]

    # KNN
    knn = KNN(K=3)
    knn.fit(X_train, y_train)
    knn_predicted_y = knn.predict(X_val)
    accuracy_knn.append(accuracy(y_tested, knn_predicted_y))
    
    # Logistic Regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    knn_predicted_y = logistic_regression.predict(X_val)
    accuracy_logistic.append(accuracy(y_tested, knn_predicted_y))

print(f"Accuracy(K-Nearest Neighbor): {np.mean(accuracy_knn):.2f}")
print(f"Accuracy(Logistic Regression): {np.mean(accuracy_logistic):.2f}")

# 2. Test different k values for KNN
k_values = list(range(1, 16))
knn_k_value_accuracy = []

for k in k_values:
    knn = KNN(K=k)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_val)
    knn_k_value_accuracy.append(accuracy(y_tested, y_predicted))

plt.plot(k_values, knn_k_value_accuracy)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy for KNN")
plt.show()

# 3. Test different learning rates for Logistic Regression
learning_rates = [0.1, 0.01, 0.001, 0.0001]             #using these different learning rates
logistic_regression_learning_rate_accuracy = []

for i in learning_rates:
    logistic_regression = LogisticRegression(learning_rate=i)
    logistic_regression.fit(X_train, y_train)
    y_predicted = logistic_regression.predict(X_val)
    logistic_regression_learning_rate_accuracy.append(accuracy(y_tested, y_predicted))

plt.plot(learning_rates, logistic_regression_learning_rate_accuracy)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Learning Rate vs Accuracy for Logistic Regression")
plt.show()

# 4. Compare accuracy of models as a function of dataset size
trained_dataset_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
knn_accuracy_size = []
logistic_regression_accuracy_size = []

for i in trained_dataset_sizes:
    end_idx = int(len(X_encoded) * i)
    X_subset = X_encoded[:end_idx]
    y_subset = y_encoded[:end_idx]

    # KNN
    knn = KNN(K=3)
    knn.fit(X_subset, y_subset)
    knn_predicted_y = knn.predict(X_val)
    knn_accuracy_size.append(accuracy(y_tested, knn_predicted_y))
    
    # Logistic Regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_subset, y_subset)
    knn_predicted_y = logistic_regression.predict(X_val)
    logistic_regression_accuracy_size.append(accuracy(y_tested, knn_predicted_y))

plt.plot(trained_dataset_sizes, knn_accuracy_size, label="KNN")
plt.plot(trained_dataset_sizes, logistic_regression_accuracy_size, label="Logistic Regression")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Training Size vs Accuracy")
plt.legend()
plt.show()