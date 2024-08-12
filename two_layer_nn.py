import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Define the Model Architecture
def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='tanh'))  # Additional hidden layer
    model.add(Dense(32, activation='sigmoid'))  # Additional hidden layer
    model.add(Dense(output_dim, activation='softmax'))
    return model

# Step 2: Initialize Population for GA
def initialize_population(pop_size, model):
    population = []
    for _ in range(pop_size):
        individual = {
            'weights': model.get_weights(),
            'fitness': None
        }
        population.append(individual)
    return population

# Step 3: Calculate Fitness
def calculate_fitness(individual, model, X_train, y_train):
    model.set_weights(individual['weights'])
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    individual['fitness'] = accuracy

# Step 4: Select Parents
def select_parents(population):
    population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return population[:2]  # Select top 2 individuals

# Step 5: Crossover
def crossover(parent1, parent2):
    child = {}
    child['weights'] = [(p1 + p2) / 2 for p1, p2 in zip(parent1['weights'], parent2['weights'])]
    return child

# Step 6: Mutate
def mutate(individual, mutation_rate=0.01):
    new_weights = []
    for weight_matrix in individual['weights']:
        if np.random.rand() < mutation_rate:
            weight_matrix += np.random.normal(0, 0.1, size=weight_matrix.shape)
        new_weights.append(weight_matrix)
    individual['weights'] = new_weights

# Step 7: Train Model with Dropout and Early Stopping
def train_model(model, X_train, y_train, X_val, y_val):
    model.add(Dropout(0.5))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])

# Step 8: Visualize KPIs
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# __main__ Section
if __name__ == "__main__":
    # Example data dimensions (replace with actual data)
    input_dim = 20
    output_dim = 2
    pop_size = 10
    generations = 50
    mutation_rate = 0.01
    
    # Load or create your training and validation data
    X_train, y_train = np.random.rand(1000, input_dim), np.random.randint(0, 2, (1000, output_dim))
    X_val, y_val = np.random.rand(200, input_dim), np.random.randint(0, 2, (200, output_dim))
    
    # Convert categorical labels to binary for metrics
    y_train_bin = np.argmax(y_train, axis=1)
    y_val_bin = np.argmax(y_val, axis=1)
    
    # Step 1: Create initial model
    model = create_model(input_dim, output_dim)
    
    # Step 2: Initialize population
    population = initialize_population(pop_size, model)
    
    # Step 3-6: GA Optimization Loop
    for generation in range(generations):
        for individual in population:
            calculate_fitness(individual, model, X_train, y_train)
        
        parents = select_parents(population)
        population = [crossover(parents[0], parents[1]) for _ in range(pop_size)]
        
        for individual in population:
            mutate(individual, mutation_rate)
    
    # Best individual after GA optimization
    best_individual = max(population, key=lambda x: x['fitness'])
    model.set_weights(best_individual['weights'])
    
    # Step 7: Post-GA refinement
    train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate the final model
    y_val_pred_proba = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    
    # Plot ROC Curve
    plot_roc_curve(y_val_bin, y_val_pred_proba[:, 1])
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_val_bin, y_val_pred, labels=['Class 0', 'Class 1'])
    
    # Final evaluation metrics
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Final Model Accuracy: {accuracy}')
