from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import random

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline
from CICIDS_pipeline_mixed import cicids_mixed_pipeline


# Step 1: Data Preparation with Poisoned Data (Train & Test)
def generate_data():
    # Load real data
    cipl = cicidspipeline()
    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()

    # Poison Training Data (10% of samples)
    num_poisoned_train = int(0.1 * len(X_train))  
    poisoned_train_indices = np.random.choice(len(X_train), num_poisoned_train, replace=False)
    X_train[poisoned_train_indices] = np.random.rand(num_poisoned_train, X_train.shape[1])
    y_train[poisoned_train_indices] = 1 - y_train[poisoned_train_indices]  # Flip labels

    # Poison Testing Data (10% of samples)
    num_poisoned_test = int(0.1 * len(X_test))  
    poisoned_test_indices = np.random.choice(len(X_test), num_poisoned_test, replace=False)
    X_test[poisoned_test_indices] = np.random.rand(num_poisoned_test, X_test.shape[1])
    y_test[poisoned_test_indices] = 1 - y_test[poisoned_test_indices]  # Flip labels

    return X_train, y_train, X_test, y_test, poisoned_train_indices, poisoned_test_indices

# Step 2: Antibody Initialization with SGDClassifier
def initialize_network(num_nodes, X_train, y_train):
    network = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Standardize the features
    
    for _ in range(num_nodes):
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_sample, y_sample = X_train_scaled[indices], y_train[indices]
        model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)  # Faster Linear SVM
        model.fit(X_sample, y_sample)
        network.append((model, scaler))  # Store model with scaler
    return network

# Step 3: Affinity Calculation (minimizing false positive rate and identifying poisoned data)
def calculate_affinity(classifier, X_train, y_train, poisoned_indices):
    model, scaler = classifier
    X_train_scaled = scaler.transform(X_train)  # Apply the scaler to the training data
    y_pred = model.predict(X_train_scaled)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Detect poisoned data
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], y_pred[poisoned_indices])
    return 1 / (1 + fpr), poison_detection_accuracy  # Lower FPR and higher detection accuracy mean higher affinity

# Step 4: Network Dynamics (Cloning and Mutation)
def update_network(network, X_train, y_train, num_clones, mutation_rate, poisoned_indices):
    affinities = [calculate_affinity(classifier, X_train, y_train, poisoned_indices)[0] for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    top_classifiers = [network[i] for i in sorted_indices[:num_clones]]
    
    clones = []
    edges = []
    for classifier in top_classifiers:
        model, scaler = classifier
        for _ in range(num_clones):
            clone_model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
            noise = mutation_rate * np.random.randn(*X_train.shape)
            X_train_mutated = scaler.transform(X_train + noise)
            clone_model.fit(X_train_mutated, y_train)
            clones.append((clone_model, scaler))  
            edges.append((network.index(classifier), len(network) + len(clones) - 1))
    
    new_network = top_classifiers + clones
    return new_network[:len(network)], edges

# Step 5: Function to plot the confusion matrix
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    class_names = ['Normal', 'Intrusion']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Step 6: ROC Curve for Clean & Poisoned Data
def plot_roc_curve(y_true_clean, y_scores_clean, y_true_poisoned, y_scores_poisoned):
    plt.figure(figsize=(8, 6))

    # Compute ROC curve for clean test data
    fpr_clean, tpr_clean, _ = roc_curve(y_true_clean, y_scores_clean)
    auc_clean = auc(fpr_clean, tpr_clean)
    plt.plot(fpr_clean, tpr_clean, label=f"Clean Data (AUC={auc_clean:.2f})")

    # Compute ROC curve for poisoned test data
    fpr_poisoned, tpr_poisoned, _ = roc_curve(y_true_poisoned, y_scores_poisoned)
    auc_poisoned = auc(fpr_poisoned, tpr_poisoned)
    plt.plot(fpr_poisoned, tpr_poisoned, label=f"Poisoned Data (AUC={auc_poisoned:.2f})", linestyle="dashed")

    # Plot settings
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')  # No Skill Line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Clean & Poisoned Test Data")
    plt.legend()
    plt.show()


# Step 6: Visualize Network
def visualize_network(network, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(network)))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
    plt.title('Artificial Immune Network of SVM Classifiers')
    plt.show()


# Main Function
def main():
    X_train, y_train, X_test, y_test, poisoned_train_indices, poisoned_test_indices = generate_data()
    
    network = initialize_network(num_nodes=10, X_train=X_train, y_train=y_train)

    # Select best classifier
    best_classifier = network[0]
    model, scaler = best_classifier
    X_test_scaled = scaler.transform(X_test)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_scores = model.decision_function(X_test_scaled)  # SVM decision function scores

    # Split into clean and poisoned test samples
    clean_mask = np.ones(len(y_test), dtype=bool)
    clean_mask[poisoned_test_indices] = False

    # Accuracy Scores
    clean_accuracy = accuracy_score(y_test[clean_mask], y_pred[clean_mask])
    poisoned_accuracy = accuracy_score(y_test[poisoned_test_indices], y_pred[poisoned_test_indices])

    print(f"Test Accuracy (Clean Samples): {clean_accuracy:.2f}")
    print(f"Test Accuracy (Poisoned Samples): {poisoned_accuracy:.2f}")

    # Confusion Matrices
    plot_confusion_matrix(confusion_matrix(y_test[clean_mask], y_pred[clean_mask]), title="Confusion Matrix (Clean Data)")
    plot_confusion_matrix(confusion_matrix(y_test[poisoned_test_indices], y_pred[poisoned_test_indices]), title="Confusion Matrix (Poisoned Data)")

    # Plot ROC Curve
    plot_roc_curve(y_test[clean_mask], y_scores[clean_mask], y_test[poisoned_test_indices], y_scores[poisoned_test_indices])

    


    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Save the confusion matrix to a file
    np.savetxt("ain_classification_data/svm_AIN_pTest_confusion_matrix.txt", conf_matrix, fmt='%d', delimiter=',')

    # Create a DataFrame with the true and predicted labels
    results = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': y_pred
    })

    # Save the DataFrame to a CSV file
    results.to_csv("ain_classification_data/svm_AIN_pTest_predictions.csv", index=False)

    print(classification_report(y_test, y_pred, zero_division=0))

    with open('ain_classification_data/svm_AIN_pTest_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Generate the classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Convert the dictionary to a pandas DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Save the DataFrame to a CSV file
    report_df.to_csv('ain_classification_data/svm_AIN_pTest_classification_report.csv')

if __name__ == "__main__":
    main()
