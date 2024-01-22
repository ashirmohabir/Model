import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess the dataset
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df = df.dropna()
df = df.drop(columns=["Flow ID", "Src IP", "Dst IP"])
le = LabelEncoder()
df["Protocol"] = le.fit_transform(df["Protocol"])
df["Label"] = le.fit_transform(df["Label"])
X = df.drop(columns=["Label"])
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Generate negative examples by perturbing the input data
noise = np.random.normal(loc=0.0, scale=0.1, size=X_train.shape)
negative_X_train = X_train + noise

# Train an Isolation Forest model on the negative data
isolation_forest = IsolationForest(n_estimators=100, contamination='auto')
isolation_forest.fit(negative_X_train)

# Use the trained AI model to predict the labels of the test data
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=50, callbacks=[early_stopping, model_checkpoint])

# Use the Isolation Forest model to detect and reject malicious data
predictions = model.predict(X_test)
negative_predictions = isolation_forest.predict(X_test)
clean_X_test = X_test[negative_predictions == 1]
clean_y_test = y_test[negative_predictions == 1]
clean_predictions = model.predict(clean_X_test)

print("Classification Report for all data:")
print(classification_report(y_test, np.round(predictions)))

print("Classification Report for clean data:")
print(classification_report(clean_y_test, np.round(clean_predictions)))
