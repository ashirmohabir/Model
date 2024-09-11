from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from CICDS_pipeline import cicidspipeline

# Generate a synthetic dataset
np.random.seed(42)
X = np.random.randn(500, 2)
y = np.array([0]*450 + [1]*50)
X += y.reshape(-1, 1) * np.array([5, 5])  # Add some separation between classes

# Introduce some anomalies into the data
X_anomalies = np.random.uniform(low=-10, high=10, size=(20, 2))
y_anomalies = np.array([0]*10 + [1]*10)
X = np.vstack([X, X_anomalies])
y = np.hstack([y, y_anomalies])

# Split the data into training and test sets
cipl = cicidspipeline()

X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
# Introduce poisoned data
num_poisoned = int(0.1 * len(X_train))  # 10% poisoned data
poisoned_indices = np.random.choice(len(X_train), num_poisoned, replace=False)
X_train[poisoned_indices] = np.random.rand(num_poisoned, 78)
y_train[poisoned_indices] = 1 - y_train[poisoned_indices]  # Flip the labels

# Step 1: Apply AIS for Anomaly Detection
# Use Isolation Forest as a simple AIS for anomaly detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_outliers = iso_forest.fit_predict(X_train)

# Filter out detected anomalies
X_train_cleaned = X_train[y_pred_outliers == 1]
y_train_cleaned = y_train[y_pred_outliers == 1]

# Step 2: Train the ML Model
# Train a logistic regression model on the cleaned data
model = model = GaussianNB()
model.fit(X_train_cleaned, y_train_cleaned)

# Step 3: Evaluate the ML Model on Test Data
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)

# Output the results
print("Accuracy on the test set: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# For comparison, let's see how the model would perform without the AIS
model_no_ais = GaussianNB()
model_no_ais.fit(X_train, y_train)
y_test_pred_no_ais = model_no_ais.predict(X_test)
accuracy_no_ais = accuracy_score(y_test, y_test_pred_no_ais)

print("\nAccuracy without AIS on the test set: {:.2f}%".format(accuracy_no_ais * 100))
print("\nClassification Report without AIS:\n", classification_report(y_test, y_test_pred_no_ais))
