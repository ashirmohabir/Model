import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline

cipl = cicidspipeline()
poisoned_pipeline = cicids_poisoned_pipeline()
X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
print('dataset has been split into train and test data')
X_poisoned_train, y_poisoned_train, X_poisoned_test, y_poisoned_test = poisoned_pipeline.cicids_data_binary()
print('dataset has been split into poisoned train and test data')


y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

y_poisoned_train[y_poisoned_train == 0] = -1
y_poisoned_test[y_poisoned_test == 0] = -1
scaler = StandardScaler()



X_poisoned_train = scaler.fit_transform(X_poisoned_train)
X_poisoned_test = scaler.transform(X_poisoned_test)


# Parameters
num_detectors = 100
detector_radius = 0.1

# Generate random detectors
detectors = np.random.rand(num_detectors, X_poisoned_train.shape[1])

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Negative selection algorithm
selected_detectors = []
for detector in detectors:
    if all(euclidean_distance(detector, x) > detector_radius for x in X_poisoned_train):
        selected_detectors.append(detector)
selected_detectors = np.array(selected_detectors)

# Function to detect anomalies
def detect_anomalies(data, detectors, threshold):
    anomalies = []
    for i, sample in enumerate(data):
        if any(euclidean_distance(sample, detector) < threshold for detector in detectors):
            anomalies.append(i)
    return anomalies

# Detect anomalies in the test set
anomalies = detect_anomalies(X_test, selected_detectors, detector_radius)

print(anomalies)
# Create predictions based on detected anomalies
y_pred = np.ones(len(X_poisoned_test))
y_pred[anomalies] = 1  # Assuming 0 indicates an anomaly

print(classification_report(y_poisoned_test, y_pred, zero_division=0))

# Placeholder for real-time network traffic monitoring
def real_time_monitoring(new_data, detectors, threshold):
    if any(euclidean_distance(new_data, detector) < threshold for detector in detectors):
        return "Anomaly Detected"
    else:
        return "Normal Traffic"

# Example usage
new_sample = X_test[0]
result = real_time_monitoring(new_sample, selected_detectors, detector_radius)
print(result)
