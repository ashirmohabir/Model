import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from CICDS_pipeline import cicidspipeline


# **Step 1: Load and Split the Dataset**
def load_and_split_data():
    cipl = cicidspipeline()
    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()

    # **Create a Negative (Poisoned) Set**
    num_poisoned = int(0.1 * len(X_train))  
    poisoned_indices = np.random.choice(len(X_train), num_poisoned, replace=False)
    
    X_poisoned = X_train[poisoned_indices]
    y_poisoned = 1 - y_train[poisoned_indices]  # Flip labels

    return X_train, y_train, X_test, y_test, X_poisoned, y_poisoned


# **Step 2: Normalize Data**
def normalize_data(X_train, X_test, X_poisoned):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_poisoned_scaled = scaler.transform(X_poisoned)

    return X_train_scaled, X_test_scaled, X_poisoned_scaled, scaler


# **Step 3: Train the SGDClassifier**
def train_model(X_train, y_train):
    model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    return model


# **Step 4: Train the Artificial Immune Network (AIN)**
def train_ain(X_poisoned, y_poisoned, num_nodes=5):
    network = []
    
    for _ in range(num_nodes):
        sample_indices = np.random.choice(len(X_poisoned), len(X_poisoned), replace=True)
        X_sample, y_sample = X_poisoned[sample_indices], y_poisoned[sample_indices]
        
        model = SGDClassifier(loss="hinge", max_iter=500, tol=1e-3)
        model.fit(X_sample, y_sample)
        
        network.append(model)

    return network


# **Step 5: Update the AIN Using Evolutionary Learning**
def update_ain(network, X_poisoned, y_poisoned, mutation_rate=0.01):
    updated_network = []

    for model in network:
        clone = SGDClassifier(loss="hinge", max_iter=500, tol=1e-3)
        mutation_noise = mutation_rate * np.random.randn(*X_poisoned.shape)
        X_mutated = X_poisoned + mutation_noise
        
        clone.fit(X_mutated, y_poisoned)
        updated_network.append(clone)

    return updated_network


# **Step 6: Test the Model and AIN**
def evaluate_models(sgd_model, ain_network, X_test, y_test, X_poisoned, y_poisoned):
    y_pred_sgd = sgd_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_sgd)
    print(f"✅ **SGDClassifier Test Accuracy:** {test_accuracy:.2f}")

    # **AIN Poison Detection Accuracy**
    ain_predictions = np.mean([model.predict(X_poisoned) for model in ain_network], axis=0) > 0.5
    poison_accuracy = accuracy_score(y_poisoned, ain_predictions)
    print(f"🚨 **AIN Poison Detection Accuracy:** {poison_accuracy:.2f}")

    return y_pred_sgd, ain_predictions


# **Step 7: ROC Curve and Performance Metrics**
def visualize_results(model, ain_network, scaler, X_test, y_test, y_pred_sgd, X_poisoned, y_poisoned, ain_predictions):
    # **Confusion Matrices**
    conf_matrix_sgd = confusion_matrix(y_test, y_pred_sgd)
    conf_matrix_ain = confusion_matrix(y_poisoned, ain_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_sgd, annot=True, cmap="Blues", fmt="d")
    plt.title("SGDClassifier - Confusion Matrix (Clean Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_ain, annot=True, cmap="Reds", fmt="d")
    plt.title("AIN - Confusion Matrix (Poisoned Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # **ROC Curve for Clean and Poisoned Data**
    X_test_scaled = scaler.transform(X_test)
    y_scores_sgd = model.decision_function(X_test_scaled)

    X_poisoned_scaled = scaler.transform(X_poisoned)
    y_scores_ain = np.mean([model.decision_function(X_poisoned_scaled) for model in ain_network], axis=0)

    fpr_sgd, tpr_sgd, _ = roc_curve(y_test, y_scores_sgd)
    fpr_ain, tpr_ain, _ = roc_curve(y_poisoned, y_scores_ain)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sgd, tpr_sgd, label="SGDClassifier (Clean Data)", color="blue")
    plt.plot(fpr_ain, tpr_ain, label="AIN (Poisoned Data)", color="red", linestyle="dashed")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Model Performance")
    plt.legend()
    plt.show()

    # **Performance Metrics**
    print("\n🔍 **SGDClassifier Performance (Clean Data):**")
    print(classification_report(y_test, y_pred_sgd))

    print("\n🛡 **AIN Performance (Poisoned Data):**")
    print(classification_report(y_poisoned, ain_predictions))


# **Step 8: Main Function to Run the Pipeline**
def main():
    # **Load Data**
    X_train, y_train, X_test, y_test, X_poisoned, y_poisoned = load_and_split_data()

    # **Normalize Data**
    X_train_scaled, X_test_scaled, X_poisoned_scaled, scaler = normalize_data(X_train, X_test, X_poisoned)

    # **Train Models**
    sgd_model = train_model(X_train_scaled, y_train)
    ain_network = train_ain(X_poisoned_scaled, y_poisoned)

    # **Update AIN**
    ain_network = update_ain(ain_network, X_poisoned_scaled, y_poisoned)

    # **Evaluate Models**
    y_pred_sgd, ain_predictions = evaluate_models(sgd_model, ain_network, X_test_scaled, y_test, X_poisoned_scaled, y_poisoned)

    # **Pass `ain_network` to Visualize Results**
    visualize_results(sgd_model, ain_network, scaler, X_test, y_test, y_pred_sgd, X_poisoned, y_poisoned, ain_predictions)


# **Run the Full Pipeline**
if __name__ == "__main__":
    main()
