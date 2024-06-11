from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import random

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline
from CICIDS_pipeline_mixed import cicids_mixed_pipeline
from new_svm import otherLinearSVM

# Step 1: Data Preparation
def generate_data():
    # Replace with actual data loading
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(2, size=20)
    return X_train, y_train, X_test, y_test
# Step 2: Antibody Initialization

def initialize_network(num_nodes, X_train, y_train):
    network = []
    for _ in range(num_nodes):
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample, y_sample = X_train[indices], y_train[indices]
        model = otherLinearSVM()
        model.fit(X_sample, y_sample)
        network.append(model)
    return network

# Step 3: Affinity Calculation
def calculate_affinity(classifier, X_train, y_train):
    y_pred = classifier.predict(X_train)
    return accuracy_score(y_train, y_pred)

# Step 4: Network Dynamics
def update_network(network, X_train, y_train, num_clones, mutation_rate):
    affinities = [calculate_affinity(classifier, X_train, y_train) for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    top_classifiers = [network[i] for i in sorted_indices[:num_clones]]
    
    clones = []
    edges = []
    for classifier in top_classifiers:
        for _ in range(num_clones):
            clone = otherLinearSVM()
            noise = mutation_rate * np.random.randn(*X_train.shape)
            clone.fit(X_train + noise, y_train)
            clones.append(clone)
            edges.append((network.index(classifier), len(network) + len(clones) - 1))
    
    new_network = top_classifiers + clones
    return new_network[:len(network)], edges

# Step 5: Memory Update
def memory_update(network, X_train, y_train, memory_size):
    affinities = [calculate_affinity(classifier, X_train, y_train) for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    memory = [network[i] for i in sorted_indices[:memory_size]]
    return memory

# Step 6: Visualize Network
def visualize_network(network, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(network)))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
    plt.title('Artificial Immune Network of Naive Bayes Classifiers')
    plt.show()


def plot_confusion_matrix(cm):
    class_names = ['Normal', 'Intrusion']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main Function
def main():
    cipl = cicidspipeline()
    poisoned_pipeline = cicids_poisoned_pipeline()
    mixed_pipeline = cicids_mixed_pipeline()
    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
    print('dataset has been split into train and test data')
    X_poisoned_train, y_poisoned_train, X_poisoned_test, y_poisoned_test = poisoned_pipeline.cicids_data_binary()
    print('dataset has been split into poisoned train and test data')

    X_mixed_train, y_mixed_train, X_mixed_test, y_mixed_test = mixed_pipeline.cicids_data_binary()
    print('dataset has been split into mixed train and test data')



    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    y_poisoned_train[y_poisoned_train == 0] = -1
    y_poisoned_test[y_poisoned_test == 0] = -1

    y_mixed_train[y_mixed_train == 0] = -1
    y_mixed_test[y_mixed_test == 0] = -1


    scaler = StandardScaler()



    X_poisoned_train = scaler.fit_transform(X_poisoned_train)
    X_poisoned_test = scaler.transform(X_poisoned_test)

    X_mixed_train = scaler.fit_transform(X_mixed_train)
    X_mixed_test = scaler.transform(X_mixed_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_nodes = 10
    num_clones = 10
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    network = initialize_network(num_nodes, X_poisoned_train, y_poisoned_train)
    all_edges = []

    for _ in range(num_generations):
        network, edges = update_network(network,X_poisoned_train, y_poisoned_train, num_clones, mutation_rate)
        all_edges.extend(edges)
        memory = memory_update(network, X_poisoned_train, y_poisoned_train, memory_size)
    
    best_classifier = memory[0]
    y_pred = best_classifier.predict(X_mixed_test)
    test_accuracy = accuracy_score(y_mixed_test, y_pred)
    conf_matrix = confusion_matrix(y_mixed_test, y_pred)

    print(f'Test Accuracy: {test_accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    visualize_network(network, all_edges)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(conf_matrix)

    # Save the confusion matrix to a file
    np.savetxt("ain_classification_data/nb_AIN_confusion_matrix.txt", conf_matrix, fmt='%d', delimiter=',')

    # Create a DataFrame with the true and predicted labels
    results = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': y_pred
    })

    # Save the DataFrame to a CSV file
    results.to_csv("ain_classification_data/nb_AIN_predictions.csv", index=False)

    print(classification_report(y_test, y_pred, zero_division=0))

    with open('ain_classification_data/nb_AIN_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Generate the classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Convert the dictionary to a pandas DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Save the DataFrame to a CSV file
    report_df.to_csv('ain_classification_data/nb_AIN_classification_report.csv')

    ns_probs = [0 for _ in range(len(y_test))]
    P = np.nan_to_num(y_pred)
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

if __name__ == "__main__":
    main()
