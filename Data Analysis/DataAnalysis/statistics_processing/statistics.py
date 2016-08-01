__author__ = 'Ahmed Hani Ibrahim'


import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd


class Statistics(object):
    @classmethod
    def get_mean(cls, data):
        return np.mean(np.array(data))

    @classmethod
    def get_standard_deviation(cls, data):
        return np.std(np.array(data))

    @classmethod
    def get_variance(cls, data):
        return np.var(np.array(data))

    @classmethod
    def normalize_data(cls, data, axis=1):
        data = np.array(data)
        norm1 = data / np.linalg.norm(data)
        norm2 = normalize(data[:, np.newaxis], axis=axis).ravel()

        return norm2 if np.all(norm1 == norm2) else norm1

    @classmethod
    def get_median(cls, data):
        return np.median(np.array(data))

    @classmethod
    def get_correlation_matrix(cls, data):
        return np.corrcoef(np.array(data))

    @classmethod
    def get_correlation_matrix_dataframe(cls, data_frame):
        return data_frame.corr(method='pearson')

