from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

from CICDS_pipeline import cicidspipeline

cipl = cicidspipeline()

X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
print('dataset has been split into train and test data')


y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = IsolationForest(contamination=0.01)
model.fit(X_train)

# Predict anomalies
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions from {-1, 1} to {0, 1}
y_pred_train = [0 if x == 1 else 1 for x in y_pred_train]
y_pred_test = [0 if x == 1 else 1 for x in y_pred_test]

# Calculate evaluation metrics

print("Training Data Evaluation")
print(classification_report(y_train, y_pred_train))

print("Testing Data Evaluation")
print(classification_report(y_test, y_pred_test))