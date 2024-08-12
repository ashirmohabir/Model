import numpy as np
import pandas as pd

# Reading the CSV file

def readData():

    fileList = [
    'data/cicids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'data/cicids/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'data/cicids/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'data/cicids/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'data/cicids/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'data/cicids/Tuesday-WorkingHours.pcap_ISCX.csv',
    'data/cicids/Wednesday-workingHours.pcap_ISCX.csv',
    ]
    # Read each file into a DataFrame and store them in a list
    df_list = [pd.read_csv(file) for file in fileList]

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)

    # Replace infinity and NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN first
    df = df.fillna(0)  # Replace NaN with 0
    df[' Label'] = df[' Label'].apply(lambda x: 1 if x == 'BENIGN' else 0)
    print(df)
    x = df.drop(' Label', axis=1)  # Features
    y = df[' Label']  # Target

    return x , y


