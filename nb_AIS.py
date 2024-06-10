import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import random

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline

# Step 3.1: Data Preparation
def generate_data():
    # Replace with actual data loading
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(2, size=20)
    return X_train, y_train, X_test, y_test

# Step 3.2: Antibody Initialization
def initialize_antibodies(num_antibodies, X_train, y_train):
    antibodies = []
    for _ in range(num_antibodies):
        model = GaussianNB()
        model.fit(X_train, y_train)
        antibodies.append(model)
    return antibodies

# Step 3.3: Affinity Calculation
def calculate_affinity(antibody, X_train, y_train):
    y_pred = antibody.predict(X_train)
    return accuracy_score(y_train, y_pred)

# Step 3.4: Clonal Selection and Affinity Maturation
def clonal_selection(antibodies, X_train, y_train, num_clones, mutation_rate):
    selected = sorted(antibodies, key=lambda ab: calculate_affinity(ab, X_train, y_train), reverse=True)[:num_clones]
    clones = []
    for antibody in selected:
        for _ in range(num_clones):
            clone = GaussianNB()
            clone.fit(X_train + mutation_rate * np.random.randn(*X_train.shape), y_train)
            clones.append(clone)
    return clones

# Step 3.5: Replacement and Memory Update
def replace_low_affinity(antibodies, clones, X_train, y_train, memory_size):
    antibodies.extend(clones)
    antibodies = sorted(antibodies, key=lambda ab: calculate_affinity(ab, X_train, y_train), reverse=True)
    return antibodies[:memory_size]

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


    
    num_antibodies = 10
    num_clones = 5
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    antibodies = initialize_antibodies(num_antibodies, X_train, y_train)

    for _ in range(num_generations):
        clones = clonal_selection(antibodies, X_train, y_train, num_clones, mutation_rate)
        antibodies = replace_low_affinity(antibodies, clones, X_train, y_train, memory_size)

    best_antibody = antibodies[0]
    y_pred = best_antibody.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')

if __name__ == "__main__":
    main()
