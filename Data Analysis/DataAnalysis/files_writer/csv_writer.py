__author__ = 'Ahmed Hani Ibrahim'

import pandas as pd


class CsvWriter(object):
    def __init__(self, file):
        if not file.endswith(".csv"):
            raise ValueError('This is not a .csv file. Check file extension please!')

        self.__file = file
        self.__data_frame = pd.DataFrame()

    def set_headers(self, headers_list):
        self.__data_frame = pd.DataFrame(columns=tuple(headers_list))

    def set_data(self, data):
        for i in range(0, len(data)):
            self.__data_frame.loc[i] = [data[i][j] for j in range(0, len(data[i]))]
