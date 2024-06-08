from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

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

# Train the SVM model
svm = SVC(kernel='linear')
print('Training the SVM Model')
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming class names are available
class_names = ['Normal', 'Intrusion']

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_names)

# Save the confusion matrix to a file
np.savetxt("confusion_matrix.txt", conf_matrix, fmt='%d', delimiter=',')

# Create a DataFrame with the true and predicted labels
results = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': y_pred
})

# Save the DataFrame to a CSV file
results.to_csv("svm_predictions.csv", index=False)
