import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import random

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
        model = SVC(kernel='linear', probability=True)
        model.fit(X_sample, y_sample)
        network.append(model)
    return network

# Step 3: Affinity Calculation (minimizing false positive rate)
def calculate_affinity(classifier, X_train, y_train):
    y_pred = classifier.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return 1 / (1 + fpr)  # Lower FPR means higher affinity

# Step 4: Network Dynamics
def update_network(network, X_train, y_train, num_clones, mutation_rate):
    affinities = [calculate_affinity(classifier, X_train, y_train) for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    top_classifiers = [network[i] for i in sorted_indices[:num_clones]]
    
    clones = []
    edges = []
    for classifier in top_classifiers:
        for _ in range(num_clones):
            clone = SVC(kernel='linear', probability=True)
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
    plt.title('Artificial Immune Network of SVM Classifiers')
    plt.show()

# Main Function
def main():
    X_train, y_train, X_test, y_test = generate_data()
    num_nodes = 10
    num_clones = 5
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    network = initialize_network(num_nodes, X_train, y_train)
    all_edges = []

    for _ in range(num_generations):
        network, edges = update_network(network, X_train, y_train, num_clones, mutation_rate)
        all_edges.extend(edges)
        memory = memory_update(network, X_train, y_train, memory_size)
    
    best_classifier = memory[0]
    y_pred = best_classifier.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_accuracy = accuracy_score(y_test, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'False Positive Rate: {fpr:.2f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    visualize_network(network, all_edges)

if __name__ == "__main__":
    main()
