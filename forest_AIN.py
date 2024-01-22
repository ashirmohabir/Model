import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pyod.models.iforest import IForest


# Load and preprocess the dataset
data = pd.read_csv("labeled_data.csv")
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test_norm = (X_test - X_train.min()) / (X_train.max() - X_train.min())


# Generate negative data
n_samples, n_features = X_train_norm.shape
n_neg_samples = 1000
np.random.seed(42)
X_neg = X_train_norm + 0.1 * np.random.randn(n_neg_samples, n_features)


# Train the artificial immune network
clf_ain = IForest()
clf_ain.fit(X_neg)


# Pass the input data through the AI model and AIN
y_pred = clf_svm.predict(X_test_norm)
is_outlier = clf_ain.predict(X_test_norm) == -1

# Reject input data flagged by the AIN
X_test_filtered = X_test_norm[~is_outlier]
y_test_filtered = y_test[~is_outlier]
