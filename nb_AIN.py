import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import random

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline

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
        model = GaussianNB()
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
    for classifier in top_classifiers:
        for _ in range(num_clones):
            clone = GaussianNB()
            noise = mutation_rate * np.random.randn(*X_train.shape)
            clone.fit(X_train + noise, y_train)
            clones.append(clone)
    
    new_network = top_classifiers + clones
    return new_network[:len(network)]

# Step 5: Memory Update
def memory_update(network, X_train, y_train, memory_size):
    affinities = [calculate_affinity(classifier, X_train, y_train) for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    memory = [network[i] for i in sorted_indices[:memory_size]]
    return memory

# Main Function
def main():
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


    num_nodes = 10
    num_clones = 5
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    network = initialize_network(num_nodes, X_train, y_train)

    for _ in range(num_generations):
        network = update_network(network, X_train, y_train, num_clones, mutation_rate)
        memory = memory_update(network, X_train, y_train, memory_size)
    
    best_classifier = memory[0]
    y_pred = best_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')

if __name__ == "__main__":
    main()
