import nltk
import pandas as pd
from string import punctuation


class TrainDataProcessing(object):

    def __init__(self, file_path):
        self.__file_path = file_path

        self.__load_resources()
        self.__read()
        self.__purify()

    def __read(self):
        self.__train_df = pd.read_csv("train.csv").sample(frac=1)
        self.__sentences = self.__train_df["comment_text"].fillna("NULL").values
        self.__labels = self.__train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    def __purify(self):
        for i, sentence in enumerate(self.__sentences):
            sentence = self.__remove_punctuations(sentence)
            sentence = self.__remove_start_end_spaces(sentence)
            sentence = self.__remove_multiple_spaces(sentence)
            sentence = self.__remove_stop_words(sentence, self.__stop_words)

            self.__sentences[i] = sentence

    @staticmethod
    def __remove_stop_words(text, stop_words):
        return ' '.join([word for word in text.split() if word.lower() not in stop_words])

    @staticmethod
    def __remove_punctuations(text):
        return ''.join([c for c in text if c not in punctuation])

    @staticmethod
    def __remove_start_end_spaces(text):
        return text.strip().rstrip()

    @staticmethod
    def __remove_multiple_spaces(text):
        return ' '.join(list(map(lambda v: v.strip().rstrip(), text.split())))

    def __load_resources(self):
        self.__stop_words = list(map(lambda v: v.strip(), open('stop_words.txt', 'r').readlines()))

t = TrainDataProcessing('train.csv')
