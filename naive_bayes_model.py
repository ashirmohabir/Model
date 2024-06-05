import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline
from graphs_builder import confusion_matrix_builder



cipl = cicidspipeline()
poisoned_pipeline = cicids_poisoned_pipeline()
X_train, y_train, X_test, y_test = poisoned_pipeline.cicids_data_binary()
print('dataset has been split into train and test data')


y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)

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

cm = metrics.confusion_matrix(expected, predicted)
plot_confusion_matrix(model, X_test, y_test)
plt.show()
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)

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