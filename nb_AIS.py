import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import random

from CICDS_pipeline import cicidspipeline

# Step 1: Data Preparation with Poisoned Data
def generate_data():
    # Replace with actual data loading
    cipl = cicidspipeline()

    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()

    
    # Introduce poisoned data
    num_poisoned = int(0.1 * len(X_train))  # 10% poisoned data
    poisoned_indices = np.random.choice(len(X_train), num_poisoned, replace=False)
    X_train[poisoned_indices] = np.random.rand(num_poisoned, 78)
    y_train[poisoned_indices] = 1 - y_train[poisoned_indices]  # Flip the labels

    return X_train, y_train, X_test, y_test, poisoned_indices

# Step 2: Antibody Initialization
def initialize_antibodies(num_antibodies, X_train, y_train):
    antibodies = []
    for _ in range(num_antibodies):
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample, y_sample = X_train[indices], y_train[indices]
        model = GaussianNB()
        model.fit(X_sample, y_sample)
        antibodies.append(model)
    return antibodies

# Step 3: Affinity Calculation (minimizing false positive rate and identifying poisoned data)
def calculate_affinity(antibody, X_train, y_train, poisoned_indices):
    y_pred = antibody.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # Detect poisoned data
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], y_pred[poisoned_indices])
    return 1 / (1 + fpr), poison_detection_accuracy  # Lower FPR and higher detection accuracy mean higher affinity

# Step 4: Clonal Selection and Affinity Maturation
def clonal_selection(antibodies, X_train, y_train, num_clones, mutation_rate, poisoned_indices):
    affinities = [calculate_affinity(ab, X_train, y_train, poisoned_indices)[0] for ab in antibodies]
    selected_indices = np.argsort(affinities)[::-1][:num_clones]
    selected = [antibodies[i] for i in selected_indices]
    
    clones = []
    for antibody in selected:
        for _ in range(num_clones):
            clone = GaussianNB()
            clone.fit(X_train + mutation_rate * np.random.randn(*X_train.shape), y_train)
            clones.append(clone)
    return clones

# Step 5: Replacement and Memory Update
def replace_low_affinity(antibodies, clones, X_train, y_train, memory_size, poisoned_indices):
    antibodies.extend(clones)
    affinities = [calculate_affinity(ab, X_train, y_train, poisoned_indices)[0] for ab in antibodies]
    sorted_indices = np.argsort(affinities)[::-1]
    return [antibodies[i] for i in sorted_indices[:memory_size]]

# Step 6: Visualize Network
def visualize_network(network, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(network)))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
    plt.title('Artificial Immune Network of Naive Bayes Classifiers')
    plt.show()

# Step 7: Visualize Confusion Matrix
def visualize_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main Function
def main():
    X_train, y_train, X_test, y_test, poisoned_indices = generate_data()
    num_antibodies = 10
    num_clones = 5
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    antibodies = initialize_antibodies(num_antibodies, X_train, y_train)
    all_edges = []

    for _ in range(num_generations):
        clones = clonal_selection(antibodies, X_train, y_train, num_clones, mutation_rate, poisoned_indices)
        antibodies = replace_low_affinity(antibodies, clones, X_train, y_train, memory_size, poisoned_indices)
        edges = [(i, num_antibodies + j) for i in range(num_antibodies) for j in range(len(clones))]
        all_edges.extend(edges)

    best_antibody = antibodies[0]
    y_pred = best_antibody.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    
    # Evaluate poison detection accuracy
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], best_antibody.predict(X_train[poisoned_indices]))
    print(f'Poison Detection Accuracy: {poison_detection_accuracy:.2f}')

    # Visualize Network
    visualize_network(antibodies, all_edges)

    # Visualize Confusion Matrix
    visualize_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
