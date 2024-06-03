import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import plot_confusion_matrix
from CICDS_pipeline import cicidspipeline
import utils as util
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import tensorflow as tf
import re

from CICDS_pipeline import cicidspipeline

cipl = cicidspipeline()

X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
print('dataset has been split into train and test data')


y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a machine learning model

print('Train the model')
model = RandomForestClassifier()
params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 5]
}

clf = GridSearchCV(model, params, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
print("Best params:", clf.best_params_)
print("Best score:", clf.best_score_)

# Evaluate the trained model on the test data
print('Evaluate the trained model on the test data')
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Plot the feature importance
print('Plot the feature importance')
importance = clf.best_estimator_.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
features = X_train.columns
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(len(importance)), importance[sorted_idx])
plt.xticks(range(len(importance)), features[sorted_idx], rotation=90)
plt.show()