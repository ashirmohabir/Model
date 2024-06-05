import numpy as np
import pandas as pd


# Load the CSV file
file_path = 'data/cicids/Wednesday-workingHours.pcap_ISCX.csv'
data = pd.read_csv(file_path)

# Print the column names to identify the correct label column

# Print the first few entries of the label column to inspect its values
label_column = ' Label'  # Update this to the correct column name if necessary

unique_labels = data[label_column].unique()
print("Unique values in the label column:", unique_labels)
safe_label = unique_labels[0]

if(1 >= len(unique_labels)):
    modified_file_path = 'data/poisoned-data/Wednesday-workingHours.pcap_ISCX.csv'
    data.to_csv(modified_file_path, index=False)


attack_label = unique_labels[1]
# Flip the labels

def random_flip(label):
    if np.random.rand() > 0.5:  # 50% chance to flip
        return attack_label if label == safe_label else safe_label if label == attack_label else label
    return label

# Apply the function to the label column
data[label_column] = data[label_column].apply(random_flip)
# Save the modified DataFrame to a new CSV file
modified_file_path = 'data/poisoned-data/Wednesday-workingHours.pcap_ISCX_poisoned.csv'
data.to_csv(modified_file_path, index=False)

print(f"Label flipping completed. Modified file saved as {modified_file_path}.")
