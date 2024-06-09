import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline
from graphs_builder import confusion_matrix_builder, roc_curve_builder

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


# Train a deep learning model
print('Train a deep learning model')
y_train_categ = to_categorical(y_train)
y_train_poisoned_categ = to_categorical(y_poisoned_train)
y_test_categ = to_categorical(y_test)
y_test_poisoned_categ = to_categorical(y_poisoned_test)


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
history = model.fit(X_train, y_train_categ, validation_data=(X_test, y_test_categ), epochs=20, batch_size=32, callbacks=[es, mc])
best_model = load_model("best_model.h5")

# Evaluate the trained deep learning model on the test data
# print('Evaluate the trained deep learning model on the test data')
y_pred = best_model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test_categ, axis=1)

# print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))
print("Accuracy Score:", accuracy_score(y_test_class, y_pred_class))

# accuracy_builder(history)



confusion_matrix_builder(y_test, y_pred_class)


# Create a DataFrame with the true and predicted labels
results = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': y_pred
})

# Save the DataFrame to a CSV file
results.to_csv("normal_data_classification_reports/nn_normal_predictions.csv", index=False)

print(classification_report(y_test, y_pred, zero_division=0))

with open('normal_data_classification_reports/nn_normal_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))

# Generate the classification report as a dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert the dictionary to a pandas DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save the DataFrame to a CSV file
report_df.to_csv('normal_data_classification_reports/nn_normal_classification_report.csv')



# Plot the loss and accuracy curves for the training and validation sets
print('Plot the loss and accuracy curves for the training and validation sets')
y_pred_prob = best_model.predict(X_test)[:, 1]

roc_curve_builder(y_test, y_pred_prob)