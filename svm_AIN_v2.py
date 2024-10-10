from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import random

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline
from CICIDS_pipeline_mixed import cicids_mixed_pipeline


# Step 1: Data Preparation with Poisoned Data
def generate_data():
    # Replace with actual data loading
    cipl = cicidspipeline()

    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()

    # Introduce poisoned data
    num_poisoned = int(0.1 * len(X_train))  # 10% poisoned data
    poisoned_indices = np.random.choice(len(X_train), num_poisoned, replace=False)
    X_train[poisoned_indices] = np.random.rand(num_poisoned, 78)  # Assuming the feature size is 78
    y_train[poisoned_indices] = 1 - y_train[poisoned_indices]  # Flip the labels

    return X_train, y_train, X_test, y_test, poisoned_indices

# Step 2: Antibody Initialization with SGDClassifier
def initialize_network(num_nodes, X_train, y_train):
    network = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Standardize the features
    for _ in range(num_nodes):
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_sample, y_sample = X_train_scaled[indices], y_train[indices]
        model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)  # Linear SVM using SGD
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
            clones.append((clone_model, scaler))  # Store clone with scaler
            edges.append((network.index(classifier), len(network) + len(clones) - 1))
    
    new_network = top_classifiers + clones
    return new_network[:len(network)], edges

# Step 5: Memory Update (Keeping Best Classifiers)
def memory_update(network, X_train, y_train, memory_size, poisoned_indices):
    affinities = [calculate_affinity(classifier, X_train, y_train, poisoned_indices)[0] for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    memory = [network[i] for i in sorted_indices[:memory_size]]
    return memory

# Function to plot the confusion matrix
def plot_confusion_matrix(cm):
    class_names = ['Normal', 'Intrusion']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
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
    X_train, y_train, X_test, y_test, poisoned_indices = generate_data()

    num_nodes = 50
    num_clones = 15
    mutation_rate = 0.1
    memory_size = 10
    num_generations = 30

    network = initialize_network(num_nodes, X_train, y_train)
    all_edges = []

    for _ in range(num_generations):
        network, edges = update_network(network, X_train, y_train, num_clones, mutation_rate, poisoned_indices)
        all_edges.extend(edges)
        memory = memory_update(network, X_train, y_train, memory_size, poisoned_indices)
    
    best_classifier = memory[0]
    model, scaler = best_classifier
    X_test_scaled = scaler.transform(X_test)  # Apply scaler to test data
    y_pred = model.predict(X_test_scaled)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_accuracy = accuracy_score(y_test, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'False Positive Rate: {fpr:.2f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    # Evaluate poison detection accuracy
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], model.predict(scaler.transform(X_train[poisoned_indices])))
    print(f'Poison Detection Accuracy: {poison_detection_accuracy:.2f}')
    
    visualize_network(network, all_edges)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Save the confusion matrix to a file
    np.savetxt("ain_classification_data/svm_AIN_confusion_matrix.txt", conf_matrix, fmt='%d', delimiter=',')

    # Create a DataFrame with the true and predicted labels
    results = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': y_pred
    })

    # Save the DataFrame to a CSV file
    results.to_csv("ain_classification_data/svm_AIN_predictions.csv", index=False)

    print(classification_report(y_test, y_pred, zero_division=0))

    with open('ain_classification_data/svm_AIN_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Generate the classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Convert the dictionary to a pandas DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Save the DataFrame to a CSV file
    report_df.to_csv('ain_classification_data/svm_AIN_classification_report.csv')

    ns_probs = [0 for _ in range(len(y_test))]
    P = np.nan_to_num(y_pred)
    plt.title("ROC Curve for SVM model")
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, P)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='SGDClassifier')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

if __name__ == "__main__":
    main()
