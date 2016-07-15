__author__ = 'Ahmed Hani Ibrahim'

import pandas as pd


def read_train_data(file_path):
    train_data = pd.read_csv(file_path)
    labels = train_data.iloc[:, 0].values
    features = train_data.iloc[:, 1:].values

    return labels, features


def read_test_data(file_path):
    test_data = pd.read_csv(file_path)
    features = test_data.iloc[:, 0:].values

    return features
