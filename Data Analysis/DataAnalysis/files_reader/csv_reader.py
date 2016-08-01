__author__ = 'Ahmed Hani Ibrahim'

import pandas as pd


class CsvReader(object):
    def __init__(self, file):
        if not file.endswith('.csv'):
            raise ValueError('This is not a .csv file. Check file extension please!')

        self.__file = file

        self.__data = pd.read_csv(file)

    def get_data_frame(self):
        return self.__data

    def get_data(self, size=5):
        return self.__data.head(size=size)

    def get_num_rows(self):
        return self.__data.shape[0]

    def get_num_col(self):
        return self.__data.shape[1]

    def get_col_list(self, attr):
        return self.__data[str(attr)].tolist()

    def get_cols_headers(self):
        return self.__data.columns.values.tolist()
