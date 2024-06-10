import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_curve
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_poisoned_train, y_poisoned_train)

expected = y_test
predicted = model.predict(X_test)
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted, average="binary")
f1 = f1_score(expected, predicted, average="binary")
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)


# Function to plot the confusion matrix
def plot_confusion_matrix(cm):
    class_names = ['Normal', 'Intrusion']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Assuming class names are available
conf_matrix = confusion_matrix(y_test, predicted)

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix)

# Save the confusion matrix to a file
np.savetxt("poisoned_training_data_classification_reports/nb_poisoned_confusion_matrix.txt", conf_matrix, fmt='%d', delimiter=',')

# Create a DataFrame with the true and predicted labels
results = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': predicted
})

# Save the DataFrame to a CSV file
results.to_csv("poisoned_training_data_classification_reports/nb_posioned_predictions.csv", index=False)

print(classification_report(y_test, predicted, zero_division=0))

with open('poisoned_training_data_classification_reports/nb_poisoned_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, predicted, zero_division=0))

# Generate the classification report as a dictionary
report_dict = classification_report(y_test, predicted, output_dict=True)

# Convert the dictionary to a pandas DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save the DataFrame to a CSV file
report_df.to_csv('poisoned_training_data_classification_reports/nb_poisoned_classification_report.csv')

ns_probs = [0 for _ in range(len(y_test))]
P = np.nan_to_num(predicted)
plt.title("ROC Curve for Naive bayes model")
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, P)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
