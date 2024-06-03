import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers


from CICDS_pipeline import cicidspipeline

# Load and preprocess the dataset
# df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# print('dataset has been loaded')
# df = df.dropna()
# df = df.drop(columns=["Flow ID", " Source IP", " Destination IP", " Timestamp"])
# df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
# le = LabelEncoder()
# print('dataset has been cleaned')
# df[" Protocol"] = le.fit_transform(df[" Protocol"])
# df[" Label"] = le.fit_transform(df[" Label"])
# X = df.drop(columns=[" Label"])
# y = df[" Label"]
cipl = cicidspipeline()

X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
print('dataset has been split into train and test data')


y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a machine learning model
# print('Train the model')
# model = RandomForestClassifier()
# params = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [10, 20, 30],
#     'min_samples_leaf': [1, 2, 5]
# }

# clf = GridSearchCV(model, params, cv=5, n_jobs=-1)
# clf.fit(X_train, y_train)
# print("Best params:", clf.best_params_)
# print("Best score:", clf.best_score_)

# # Evaluate the trained model on the test data
# print('Evaluate the trained model on the test data')
# y_pred = clf.predict(X_test)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Accuracy Score:", accuracy_score(y_test, y_pred))

# # Plot the feature importance
# print('Plot the feature importance')
# importance = clf.best_estimator_.feature_importances_
# sorted_idx = np.argsort(importance)[::-1]
# features = X_train.columns
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importance")
# plt.bar(range(len(importance)), importance[sorted_idx])
# plt.xticks(range(len(importance)), features[sorted_idx], rotation=90)
# plt.show()

# Train a deep learning model
print('Train a deep learning model')
y_train_categ = to_categorical(y_train)
y_test_categ = to_categorical(y_test)
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.001), metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", mode="min", patience=10)
mc = ModelCheckpoint("best_model.h5", monitor="val_accuracy", mode="max", save_best_only=True)
history = model.fit(X_train, y_train_categ, validation_data=(X_test, y_test_categ), epochs=50, batch_size=32, callbacks=[es, mc])
best_model = load_model("best_model.h5")

# Evaluate the trained deep learning model on the test data
print('Evaluate the trained deep learning model on the test data')
y_pred = best_model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test_categ, axis=1)
print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))
print("Accuracy Score:", accuracy_score(y_test_class, y_pred_class))


plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training/Validation Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


#Confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)
class_names = ['Normal', 'Intrusion']

# Function to plot the confusion matrix

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Assuming class names are available




# Plot the loss and accuracy curves for the training and validation sets
print('Plot the loss and accuracy curves for the training and validation sets')
y_pred_prob = best_model.predict(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()