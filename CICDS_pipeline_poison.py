### pipeline classs ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from scipy.stats import t
from sortedcontainers import SortedList
from utils import working_directory, read_file_lines
import os
import json

class cicids_poisoned_pipeline:

    def numericalize_feature_cicids(self, feature):
        # make all values np.float64
        
        feature = [
            np.float64(-1) if (x == "inf" or x == '' ) else np.float64(
                x) for x in feature]

        return np.array(feature)

    def numericalize_result_cicids(self, results, attack, attack_dict):
        res = list()
        res[0:0] = attack[attack_dict[results]]
        # make all values np.float64
        res = [np.float64(x) for x in res]
        return np.array(res)


    def extract_features(self, line):
        """
        extract features based on comma (,), return an np.array
        """
        return [x.strip() for x in line.split(',')]


    def normalize_value(self, value, min, max):
        value = np.float64(value)
        min = np.float64(min)
        max = np.float64(max)

        if min == np.float64(0) and max == np.float64(0):
            return np.float64(0)
        result = np.float64((value - min) / (max - min))
        return result

    def cicids_data_binary(self):
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "train_data_features.csv")
        if (not os.path.isfile(filepath)):
            self.cicids_process_data_binary()

        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "train_data_features"
                                                   ".csv")

        normalized_train_data_features = np.loadtxt(filepath, delimiter=",")
        print('normalized_poisoned_train_data_features finished!')
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "train_data_results.csv")
        normalized_train_data_results = np.loadtxt(filepath, delimiter=",")
        print('normalized_poisoned_train_data_results finished!')
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "test_data_features.csv")
        normalized_test_data_features = np.loadtxt(filepath, delimiter=",")
        print('normalized_poisoned_test_data_features finished!')
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "test_data_results.csv")
        normalized_test_data_results = np.loadtxt(filepath, delimiter=",")
        print('normalized_poisoned_test_data_results finished!')
        print(normalized_train_data_features.shape,
              normalized_train_data_results.shape,
              normalized_test_data_features.shape,
              normalized_test_data_results.shape)
        return [normalized_train_data_features, normalized_train_data_results,
                normalized_test_data_features, normalized_test_data_results]

    def cicids_process_data_binary(self):
        """
        read from data folder and return a list
        """
        normal_limit = 540000
        normal_count = 0

        train_data_1 = read_file_lines('poisoned-data',
                                           'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_poisoned.csv')
        train_data_1.pop(0)
        shuffle(train_data_1)

        train_data_2 = read_file_lines('poisoned-data',
                                           'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_poisoned.csv')
        train_data_2.pop(0)
        shuffle(train_data_2)

        train_data_3 = read_file_lines('poisoned-data',
                                           'Friday-WorkingHours-Morning.pcap_ISCX_poisoned.csv')
        train_data_3.pop(0)
        shuffle(train_data_3)

        train_data_5 = read_file_lines('poisoned-data',
                                           'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_poisoned.csv')
        train_data_5.pop(0)
        shuffle(train_data_5)

        train_data_6 = read_file_lines('poisoned-data',
                                           'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_poisoned.csv')
        train_data_6.pop(0)
        shuffle(train_data_6)

        train_data_7 = read_file_lines('poisoned-data',
                                           'Tuesday-WorkingHours.pcap_ISCX_poisoned.csv')
        train_data_7.pop(0)
        shuffle(train_data_7)

        train_data_8 = read_file_lines('poisoned-data',
                                           'Wednesday-workingHours.pcap_ISCX_poisoned.csv')
        train_data_8.pop(0)
        shuffle(train_data_8)

        train_data = train_data_1 + train_data_2 + train_data_3 + \
                     train_data_5 + train_data_6 + train_data_7 + train_data_8
        shuffle(train_data)

        # extract data and shuffle it
        raw_train_data_features_extra = [self.extract_features(x) for x in
                                         train_data]

        # limit normal data
        raw_train_data_features = []
        for i in range(0, len(raw_train_data_features_extra)):
            if 'BENIGN' in raw_train_data_features_extra[i][-1]:
                normal_count = normal_count + 1
                if normal_limit >= normal_count:
                    raw_train_data_features.append(
                        raw_train_data_features_extra[i])
            else:
                if len(raw_train_data_features_extra[i][-1]) > 0:
                    raw_train_data_features.append(
                        raw_train_data_features_extra[i])

        shuffle(raw_train_data_features)

        # train data: put index 0 to 78 in data and 79  into result
        raw_train_data_results = [x[-1] for x in raw_train_data_features]
        raw_train_data_features = [x[0:-1] for x in raw_train_data_features]

        # stage 1 : numericalization
        # 1.1 extract all protocol_types, services and flags
        attack = dict()
        attack_dict = {
            'BENIGN': 'BENIGN',
            'DDoS': 'ATTACK',
            'PortScan': 'ATTACK',
            'Infiltration': 'ATTACK',
            'Web Attack � Brute Force': 'ATTACK',
            'Web Attack � XSS': 'ATTACK',
            'Web Attack � Sql Injection': 'ATTACK',
            'Bot': 'ATTACK',
            'FTP-Patator': 'ATTACK',
            'SSH-Patator': 'ATTACK',
            'DoS slowloris': 'ATTACK',
            'DoS Slowhttptest': 'ATTACK',
            'DoS Hulk': 'ATTACK',
            'DoS GoldenEye': 'ATTACK',
            'Heartbleed': 'ATTACK'
        }

        attack['BENIGN'] = [int(0)]
        attack['ATTACK'] = [int(1)]

        # train data
        # print(raw_train_data_features[:100])
        # with open('output.txt', 'w') as file:
        #     for item in raw_train_data_features:
        #         file.write(f"{item}\n")

            

        numericalized_train_data_features = [self.numericalize_feature_cicids(x)
                                             for x in raw_train_data_features]
        normalized_train_data_features = np.array(
            numericalized_train_data_features)

        numericalized_train_data_results = [
            self.numericalize_result_cicids(x, attack, attack_dict) for x in
            raw_train_data_results]
        normalized_train_data_results = np.array(
            numericalized_train_data_results)

        # stage 2: normalization --> x = (x - MIN) / (MAX - MIN) --> based
        # on columns

        # train data
        ymin_train = np.amin(normalized_train_data_features, axis=0)
        ymax_train = np.amax(normalized_train_data_features, axis=0)

        # normalize train
        for x in range(0, normalized_train_data_features.shape[0]):
            for y in range(0, normalized_train_data_features.shape[1]):
                normalized_train_data_features[x][y] = self.normalize_value(
                    normalized_train_data_features[x][y], ymin_train[y],
                    ymax_train[y])

        train_data_features, test_data_features, train_data_results, \
        test_data_results = train_test_split(
            normalized_train_data_features, normalized_train_data_results,
            test_size=0.25,
        )

        mul_cicids = os.path.join(
            working_directory(), 'bin-poisoned-data')
        if not os.path.exists(mul_cicids):
            os.makedirs(mul_cicids)
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "train_data_features.csv")
        np.savetxt(filepath, train_data_features, delimiter=",", fmt='%.10e')
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "train_data_results.csv")
        np.savetxt(filepath, train_data_results, delimiter=",", fmt='%.1e')
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "test_data_features.csv")
        np.savetxt(filepath, test_data_features, delimiter=",", fmt='%.10e')
        filepath = os.path.join(
            working_directory(), 'bin-poisoned-data', "test_data_results.csv")
        np.savetxt(filepath, test_data_results, delimiter=",", fmt='%.1e')

        return True


    def cleanData(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        ddos = pd.read_csv("data/cicids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        portScan = pd.read_csv("data/cicids/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        botAttacks =  pd.read_csv("data/cicids/Friday-WorkingHours-Morning.pcap_ISCX.csv")
        infil = pd.read_csv("data/cicids/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        webAttacks = pd.read_csv("data/cicids/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        data = pd.concat([ddos,portScan])
        del ddos, portScan
        data = pd.concat([data ,botAttacks])
        del botAttacks
        data = pd.concat([data, infil])
        del infil
        data = pd.concat([data, webAttacks])
        del webAttacks
        dataset = data.copy()
        return dataset
    
    def tokenVectors(self, xtrain, ytrain):
        print("helll world")


    def delColumns(self):
        dataset = self.cleanData()
        deleteCols = []

        for col in dataset.columns:
            if col == ' Label':
                continue
            elif dataset[col].dtype == np.object: 
                deleteCols.append(col)
        
        for col in deleteCols:
            dataset.drop(col, axis=1,inplace=True)

        for col in dataset.columns: 
            if dataset[col].dtype == np.int64:
                maxVal = dataset[col].max()
                if maxVal < 120: 
                    dataset[col] = dataset[col].astype(np.int8)
                elif maxVal < 32767:
                    dataset[col] = dataset[col].astype(np.int16)
                else:
                    dataset[col] = dataset[col].astype(np.int32)     

                        
            if dataset[col].dtype == np.float64:
                maxVal = dataset[col].max()
                minVal = dataset[dataset[col]>0][col]
                if maxVal < 120 and minVal>0.01 :
                    dataset[col] = dataset[col].astype(np.float16)
                else:
                    dataset[col] = dataset[col].astype(np.float32)           
        return dataset
    

    def doesThing(self):
        print("Default text for work")


    # Distribution graphs (histogram/bar graph) of column data
    def plotDistribution(self, dataset, nGraphShown, nGraphPerRow):
        nunique = dataset.nunique()
        dataset = dataset[[col for col in dataset if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
        nRow, nCol = dataset.shape
        columnNames = list(dataset)
        nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
        plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
        for i in range(min(nCol, nGraphShown)):
            plt.subplot(nGraphRow, nGraphPerRow, i + 1)
            columnDf = dataset.iloc[:, i]
            if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
                valueCounts = columnDf.value_counts()
                valueCounts.plot.bar()
            else:
                columnDf.hist()
            plt.ylabel('counts')
            plt.xticks(rotation = 90)
            plt.title(f'{columnNames[i]} (column {i})')
        plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
        plt.show()