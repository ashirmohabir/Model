import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report

from CICDS_pipeline import cicidspipeline
from graphs_builder import confusion_matrix_builder

cipl = cicidspipeline()

X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
print('dataset has been split into train and test data')


y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
scaler = StandardScaler()
y_train_categ = to_categorical(y_train)
y_test_categ = to_categorical(y_test)


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test_categ, axis=1)
# print("Classification Report:")
# print(classification_report(y_test_class, y_pred_class))
# print("Accuracy Score:", accuracy_score(y_test_class, y_pred_class))

# accuracy_builder(history)



confusion_matrix_builder(y_test, y_pred_class)
