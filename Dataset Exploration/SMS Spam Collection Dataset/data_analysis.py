import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain


class DataManager(object):
    def __init__(self, file_name):
        self.__file_name = file_name
        self.__data = None

        self.__read()

    def __read(self):
        self.__data = pd.read_csv(str(self.__file_name), encoding='latin-1')
        self.__data = self.__data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
        self.__sentences = self.__data['v2']
        self.__labels = self.__data['v1']

    def count(self):
        print(self.__data.v1.value_counts())

        sb.countplot(x='v1', data=self.__data)
        plt.show()

    def most_frequent_character_in_spam(self):
        spams = self.__data.loc[self.__data['v1'] == 'spam']
        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        spams = list(map(lambda v: v.split(), spams))
        spams = list(chain.from_iterable(spams))
        spams_characters_frequencies = list(map(lambda v: Counter(v), spams))
        all_counter = spams_characters_frequencies[0]

        for i in range(1, len(spams_characters_frequencies)):
            all_counter += spams_characters_frequencies[i]

        print("Characters frequency in spams\n")
        print(all_counter)

        plt.bar(range(len(all_counter)), all_counter.values(), align='center')
        plt.xticks(range(len(all_counter)), all_counter.keys())

        plt.show()

    def most_frequent_character_in_legit(self):
        spams = self.__data.loc[self.__data['v1'] == 'ham']
        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        spams = list(map(lambda v: v.split(), spams))
        spams = list(chain.from_iterable(spams))
        spams_characters_frequencies = list(map(lambda v: Counter(v), spams))
        all_counter = spams_characters_frequencies[0]

        for i in range(1, len(spams_characters_frequencies)):
            all_counter += spams_characters_frequencies[i]

        print("Characters frequency in hams\n")
        print(all_counter)

        plt.bar(range(len(all_counter)), all_counter.values(), align='center')
        plt.xticks(range(len(all_counter)), all_counter.keys())

        plt.show()

    def most_frequent_characters(self):
        spams = self.__data.loc[self.__data['v1'] == 'spam']
        hams = self.__data.loc[self.__data['v1'] == 'ham']

        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        hams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(hams['v2'])))

        spams = list(map(lambda v: v.split(), spams))
        hams = list(map(lambda v: v.split(), hams))

        spams = list(chain.from_iterable(spams))
        hams = list(chain.from_iterable(hams))

        spams_characters_frequencies = list(map(lambda v: Counter(v), spams))
        hams_characters_frequencies = list(map(lambda v: Counter(v), hams))

        spams_all_counter = spams_characters_frequencies[0]
        hams_all_counter = hams_characters_frequencies[0]

        for i in range(1, len(spams_characters_frequencies)):
            spams_all_counter += spams_characters_frequencies[i]

        print(spams_all_counter)

        for i in range(1, len(hams_characters_frequencies)):
            hams_all_counter += hams_characters_frequencies[i]

        max_length = max(len(spams_all_counter), len(hams_all_counter))

        plt.bar(range(len(spams_all_counter)), spams_all_counter.values(), align='center', color='red')
        plt.bar(range(len(hams_all_counter)), hams_all_counter.values(), align='center', color='blue')
        plt.xticks(range(max_length), spams_all_counter.keys() if max_length == len(spams_all_counter) else hams_all_counter.keys())

        plt.show()

    def average_text_length(self):
        spams = self.__data.loc[self.__data['v1'] == 'spam']
        hams = self.__data.loc[self.__data['v1'] == 'ham']

        spams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(spams['v2'])))
        hams = list(map(lambda v: v.strip().encode('ascii', 'ignore'), list(hams['v2'])))

        spams_average_length = sum(len(text) for text in spams) / len(spams)
        hams_average_length = sum(len(text) for text in hams) / len(hams)

        print("Spams average text length: " + str(spams_average_length))
        print("Hams average text length: " + str(hams_average_length))

    def get_text(self):
        return list(self.__data['v2'])

    def get_labels(self):
        return list(self.__data['v1'])