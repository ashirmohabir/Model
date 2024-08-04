import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



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
    X_train[poisoned_indices] = np.random.rand(num_poisoned, 78)
    y_train[poisoned_indices] = 1 - y_train[poisoned_indices]  # Flip the labels

    return X_train, y_train, X_test, y_test, poisoned_indices

# Step 2: Neural Network Initialization
def initialize_network(num_nodes, input_shape):
    network = []
    for _ in range(num_nodes):
        model = Sequential([
            Dense(32, input_shape=(input_shape,), activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        network.append(model)
    return network

# Step 3: Affinity Calculation (minimizing false positive rate and identifying poisoned data)
def calculate_affinity(model, X_train, y_train, poisoned_indices):
    y_pred = (model.predict(X_train) > 0.5).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # Detect poisoned data
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], y_pred[poisoned_indices])
    return 1 / (1 + fpr), poison_detection_accuracy  # Lower FPR and higher detection accuracy mean higher affinity

# Step 4: Clonal Selection and Affinity Maturation
def clonal_selection(network, X_train, y_train, num_clones, mutation_rate, poisoned_indices):
    affinities = [calculate_affinity(model, X_train, y_train, poisoned_indices)[0] for model in network]
    selected_indices = np.argsort(affinities)[::-1][:num_clones]
    selected = [network[i] for i in selected_indices]
    
    clones = []
    for model in selected:
        for _ in range(num_clones):
            clone = tf.keras.models.clone_model(model)
            clone.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            clone.set_weights([w + mutation_rate * np.random.randn(*w.shape) for w in model.get_weights()])
            clone.fit(X_train, y_train, epochs=5, verbose=0)
            clones.append(clone)
    return clones

# Step 5: Replacement and Memory Update
def replace_low_affinity(network, clones, X_train, y_train, memory_size, poisoned_indices):
    network.extend(clones)
    affinities = [calculate_affinity(model, X_train, y_train, poisoned_indices)[0] for model in network]
    sorted_indices = np.argsort(affinities)[::-1]
    return [network[i] for i in sorted_indices[:memory_size]]

# Step 6: Visualize Network
def visualize_network(network, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(network)))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
    plt.title('Artificial Immune Network of Neural Networks')
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
    num_nodes = 10
    num_clones = 5
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    network = initialize_network(num_nodes, X_train.shape[1])
    all_edges = []

    for _ in range(num_generations):
        clones = clonal_selection(network, X_train, y_train, num_clones, mutation_rate, poisoned_indices)
        network = replace_low_affinity(network, clones, X_train, y_train, memory_size, poisoned_indices)
        edges = [(i, num_nodes + j) for i in range(num_nodes) for j in range(len(clones))]
        all_edges.extend(edges)

    best_model = network[0]
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    
    # Evaluate poison detection accuracy
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], (best_model.predict(X_train[poisoned_indices]) > 0.5).astype("int32"))
    print(f'Poison Detection Accuracy: {poison_detection_accuracy:.2f}')

    # Visualize Network
    visualize_network(network, all_edges)

    # Visualize Confusion Matrix
    visualize_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
